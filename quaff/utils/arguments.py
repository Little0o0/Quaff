from dataclasses import dataclass, field
from typing import Optional
import transformers
import os
import argparse

@dataclass
class ModelArguments:
    # model_name_or_path: Optional[str] = field(
    #     default="EleutherAI/pythia-12b"
    # )

    model_name_or_path: Optional[str] = field(
            default="facebook/opt-125m"
        )

    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: Optional[float] = field(
        default=0.2, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset_name: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=False,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    peft_type: str = field(
        default="lora",
        metadata={"help": "the type of fine tuning method, e.g. `lora`, `prefix`, ``"}
    )
    num_virtual_tokens: int = field(
        default=20,
        metadata={"help": "How many num_virtual_tokens to use in prefix and prompt tuning."}
    )
    token_dim: int = field(
        default=None,
        metadata={"help": "How many token_dim to use in prefix and prompt tuning.."}
    )
    prefix_hidden_size: int = field(
        default=20,
        metadata={"help": "How many encoder_hidden_size to use in prefix_tuning."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    lora_modules: str = field(
        default= "all" ,
        metadata={"help":"Lora modules, str split by '|' or 'all' ."}
    )
    max_memory_MB: int = field(
        default=32000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='wandb',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    checkpoint_dir: str = field(default='./output', metadata={"help": 'The input dir for checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=-1, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=False, metadata={"help": 'To train or not to train, that is the question?'})
    do_eval: bool = field(default=False, metadata={"help": 'To eval or not to train, that is the question?'})
    do_calibration: bool = field(default=False, metadata={"help": 'if generate outlier idx ?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    logging_strategy: str = field(default="steps")
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=1, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    grad_precision: Optional[int] = field(default=8)
    weight_quant_type: str = field(
        default='vector',
        metadata={"help": "Which quant algorithm is used to quant model weight: linear, vector"}
    )
    act_quant_type: str = field(
        default='vector',
        metadata={"help": "Which quant algorithm is used to quant activation: linear, vector"}
    )
    cal_dataset_name: str = field(
        default='chip2',
        metadata={
            "help": "Which oasst1 is used to generate outlier channels and scaling factor. See data_module for options."}
    )
    cal_dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )
    max_cal_samples: Optional[int] = field(default=512)
    outlier_strategy: str = field(
        default='mean_times',
        metadata={
            "help": "Which strategy is used to generate outlier channels. mean_times, static, zscore, top_abs"}
    )

    outlier_threshold: float = field(
        default=100.,
        metadata={
            "help": "if using static outlier strategy, it should set the outlier threshold."}
    )
    
    outlier_output_dir: str = field(
        default=f'./outlier/',
        metadata={
            "help": "the root dir to store outlier"}
    )
    outlier_input_dir: str = field(
        default=f'./outlier/',
        metadata={
            "help": "the root dir to load outlier."}
    )
    
    S_momentum: bool = field(default=False)
    outlier_max_ratio: Optional[float] = field(default=0.01)
    seed: int = field(default=0,)
    num_train_epochs: int = field(default=1,)

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)
    
def get_args():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    model_name = model_args.model_name_or_path.split("/")[-1]
    # determine the detailed saving dir
    training_args.output_dir = os.path.join(training_args.output_dir, model_name, data_args.dataset_name,training_args.peft_type, training_args.lora_modules)
    training_args.checkpoint_dir = os.path.join(training_args.checkpoint_dir, model_name, data_args.dataset_name, training_args.peft_type, training_args.lora_modules)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    training_args.run_name = f"{training_args.peft_type}-{model_name}-{data_args.dataset_name}"

    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args), **vars(generation_args)
    )

    return args, training_args