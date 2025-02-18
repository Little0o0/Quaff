import os
from os.path import exists, join, isdir
from typing import Dict
import bitsandbytes as bnb
import importlib
from packaging import version
import warnings
import torch
from torch import nn
import transformers
from transformers import (
    GPTNeoXForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
    IA3Config,
    get_peft_model,
    PeftModel
)

from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from quaff.nn.module import Linear8bitQuaff
from quaff.utils.alignment import peft_name_to_model_name

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def find_all_linear_layer_paths(args, model):
    cls = torch.nn.Linear
    linear_layer_paths = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            linear_layer_paths.add(name)
    return linear_layer_paths

def find_all_linear_names(args, model):
    linear_layer_paths = find_all_linear_layer_paths(args, model)
    lora_module_names = set()
    for path in linear_layer_paths:
        path = path.split('.')
        module_name = path[0] if len(path) == 1 else path[-1]
        lora_module_names.add(path[0] if len(path) == 1 else path[-1])

    if 'lm_head' in lora_module_names: # do not need to be quant
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False

def build_quantized_model(model, outlier_dict, grad_precision=8, weight_quant_type="linear",act_quant_type="vector", S_momentum=True):
    def _get_submodules(model, key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    keylist = [ key for key, module in model.named_modules() if isinstance(module, nn.Linear)]
    # replace all linear layer to quantized int8 layer
    for key in keylist:
        if "lm_head" in key:
            # ignore the last linear layer
            continue
        parent, target, target_name = _get_submodules(model, key)
        new_module = quantize_linear_layer(target, key, outlier_dict=outlier_dict, grad_precision=grad_precision, weight_quant_type=weight_quant_type, act_quant_type=act_quant_type, S_momentum=S_momentum)
        setattr(parent, target_name, new_module)
    return model

def quantize_linear_layer(module, layer_name, outlier_dict, grad_precision=8, weight_quant_type="linear",act_quant_type="vector", S_momentum=True):
    new_module = module
    assert outlier_dict is not None
    outlier_idx = None if layer_name not in outlier_dict else outlier_dict[layer_name]
    if outlier_idx is None:
        warnings.warn(f"{layer_name} is not included in predefined outlier_dict !!!! ")
    new_module = Linear8bitQuaff.load_from_linear(module, outlier_idx, S_momentum=S_momentum, weight_quant_type=weight_quant_type, act_quant_type=act_quant_type, grad_precision=grad_precision, layer_name=layer_name)
    return new_module


def build_model(args, ):
    # in this function, we need to produce a full_precision or quantized model, from init or pretrained, based on setting.
    # we also need to set the computation type.

    # Determine GPUs
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
    max_memory = f'{args.max_memory_MB}MB'

    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype =  torch.float32
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=compute_dtype,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token
    )

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = compute_dtype

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast= True if "pythia" in args.model_name_or_path else False,  # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None,  # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
            ),
        })

    # quantize the model
    return model, tokenizer


def build_peft_model(args, model, tokenizer):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    model.enable_input_require_grads()
    
    if args.peft_type == "lora" :
        print(f'adding LoRA modules...')
        if args.lora_modules == "all":
            modules = find_all_linear_names(args, model)
        else:
            modules = args.lora_modules.split("|")
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules= modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            inference_mode=False,
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    elif args.peft_type == "ia3" :
        print(f'adding IA3 modules...')
        if args.lora_modules == "all":
            modules = find_all_linear_names(args, model)
        else:
            modules = args.lora_modules.split("|")
        config = IA3Config(
            task_type="CAUSAL_LM",
            target_modules=modules, 
            inference_mode=False,
            feedforward_modules = modules,
        )
        model = get_peft_model(model, config)

    elif args.peft_type == "prefix" :
        print(f'adding Prefix modules...')
        # modules = find_all_linear_names(args, model)
        config = PrefixTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=args.num_virtual_tokens,
            # base_model_name_or_path=args.model_name_or_path,
            # token_dim = args.token_dim,
            encoder_hidden_size = args.prefix_hidden_size,
            inference_mode=False,
        )
        model = get_peft_model(model, config)

    elif args.peft_type == "prompt" :
        print(f'adding Prompt modules...')
        # modules = find_all_linear_names(args, model)
        config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=args.num_virtual_tokens,
            token_dim = args.token_dim,
            inference_mode=False,
        )
        model = get_peft_model(model, config)

    elif args.peft_type == "ptuning" :
        print(f'adding P-Tuning modules...')
        # modules = find_all_linear_names(args, model)
        config = PromptEncoderConfig(
            task_type="CAUSAL_LM",
            encoder_reparameterization_type="MLP",
            num_virtual_tokens=args.num_virtual_tokens,
            token_dim = args.token_dim,
            inference_mode=False,
        )
        model = get_peft_model(model, config)
    elif args.peft_type == "none":
        pass
    else:
        raise Exception(f"Does not support {args.peft_type} peft method yet")

    return model, tokenizer
