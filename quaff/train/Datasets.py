from datasets import load_dataset, Dataset
import pandas as pd
import os
import transformers
from typing import Optional, Dict, Sequence
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
import torch
import copy
import random
import string

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict



def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

GPQA_PROMPT = (
    "{} Please select one of the following options: " 
    "(A) {}. (B) {}. (C) {}. (D) {}."
)
abcd = abc = [f"({c})" for c in string.ascii_uppercase]
GPQA_ANSWER = "{} The answer is {}."

def build_gpqa_data(example):
    question = example["Question"]
    answer = example["Explanation"]
    options = [0, 1, 2, 3]
    random.shuffle(options)
    choices = ["", "", "", ""]
    correct_option = abcd[options[0]]
    choices[options[0]] = example["Correct Answer"]
    choices[options[1]] = example["Incorrect Answer 1"]
    choices[options[2]] = example["Incorrect Answer 2"]
    choices[options[3]] = example["Incorrect Answer 3"]
    input_text = GPQA_PROMPT.format(question, *choices)
    output_text = GPQA_ANSWER.format(answer, correct_option)
    return {'input': input_text, 'output': output_text}

def build_mmlupro_data(example):
    abc = [f"({c})" for c in string.ascii_uppercase]
    PROMPT = "{} Please select one of the following options: "
    options = example["options"]
    question = example["question"]
    answer = "The answer is ({})".format(example["answer"])
    input_text = PROMPT.format(question)
    for idx, op in enumerate(options):
        input_text += f"{abc[idx]} {op} "
    return {'input': input_text, 'output': answer}

def build_mathqa_data(example):
    PROMPT = "{} Please select one of the following options: {}"
    question = example["Problem"]
    options = example["options"]
    for c in string.ascii_lowercase:
        if f"{c} )" in options:
            options = options.replace(f"{c} )", f"({c.upper()})")
    input_text = PROMPT.format(question, options)
    correct = example["correct"].upper()
    answer = "answer".join(example["Rationale"].split("answer")[:-1])
    answer = f"{answer}. The answer is ({correct}).\""
    return {'input': input_text, 'output': answer}

def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

def load_data(dataset_name, cache_dir=""):
    if dataset_name == 'alpaca':
        return load_dataset("tatsu-lab/alpaca")
    elif dataset_name == 'alpaca-clean':
        return load_dataset("yahma/alpaca-cleaned")
    elif dataset_name == "finance-alpaca":
        return load_dataset("gbharti/finance-alpaca")
    elif dataset_name == 'chip2':
        return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
    elif dataset_name == 'self-instruct':
        return load_dataset("yizhongw/self_instruct", name='self_instruct')
    elif dataset_name == 'hh-rlhf':
        return load_dataset("Anthropic/hh-rlhf")
    elif dataset_name == 'longform':
        return load_dataset("akoksal/LongForm")
    elif dataset_name == 'oasst1':
        return load_dataset("timdettmers/openassistant-guanaco")
    elif dataset_name == "wikitext2":
        return load_dataset("wikitext", "wikitext-2-raw-v1")
    elif dataset_name == "lambada":
        return load_dataset("EleutherAI/lambada_openai", "en")
    elif dataset_name == "gsm8k":
        return load_dataset("openai/gsm8k", "main")
    elif dataset_name == "mmlupro":
        return load_dataset("TIGER-Lab/MMLU-Pro")
    elif dataset_name == "gpqa":
        return load_dataset("Idavidrein/gpqa", "gpqa_main")
    elif dataset_name == "mathqa":
        return load_dataset("allenai/math_qa")
    elif dataset_name == 'vicuna':
        raise NotImplementedError("Vicuna data was not released.")
    else:
        if os.path.exists(dataset_name):
            try:
                full_dataset = local_dataset(dataset_name)
                return full_dataset
            except:
                raise ValueError(f"Error loading dataset from {dataset_name}")
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

def format_dataset(dataset, dataset_format, dataset_name):
    if (
        dataset_format in ['alpaca', 'alpaca-clean', "finance-alpaca"] or
        (dataset_format is None and dataset_name in ['alpaca', 'alpaca-clean', "finance-alpaca"])
    ):
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
    elif dataset_format == "lambada" or (dataset_format is None and dataset_name == 'lambada'):
        dataset = dataset.map(lambda x: {
            'input': '',
            'output': x['text']
        })
        if "train" not in dataset and "test" in dataset:
            dataset["train"] = dataset["test"]
            del dataset["test"]
    elif dataset_format == "gsm8k" or (dataset_format is None and dataset_name == 'gsm8k'):
        ## prompt or not ?
        dataset = dataset.map(lambda x: {
            'input': x['question'],
            'output': x['answer']
        }, remove_columns=['question', 'answer'])
        
    elif dataset_format == "gpqa" or (dataset_format is None and dataset_name == 'gpqa'):
        dataset = dataset.select_columns(["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Explanation"])
        dataset = dataset.map(build_gpqa_data)

    elif dataset_format == "mmlupro" or (dataset_format is None and dataset_name == 'mmlupro'):
        dataset["train"] = dataset["test"]
        del dataset["test"]
        dataset = dataset.select_columns(["question", "options", "answer"])
        dataset = dataset.map(build_mmlupro_data)
    
    elif dataset_format == "mathqa" or (dataset_format is None and dataset_name == 'mathqa'):
            dataset = dataset.select_columns(["Problem", "options", "Rationale", "correct"])
            dataset = dataset.map(build_mathqa_data)

    elif dataset_format == 'chip2' or (dataset_format is None and dataset_name == 'chip2'):
        dataset = dataset.map(lambda x: {
            'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
            'output': x['text'].split('\n<bot>: ')[1],
        })
    elif dataset_format == 'self-instruct' or (dataset_format is None and dataset_name == 'self-instruct'):
        for old, new in [["prompt", "input"], ["completion", "output"]]:
            dataset = dataset.rename_column(old, new)
    elif dataset_format == 'hh-rlhf' or (dataset_format is None and dataset_name == 'hh-rlhf'):
        dataset = dataset.map(lambda x: {
            'input': '',
            'output': x['chosen']
        })
    elif dataset_format == 'oasst1' or (dataset_format is None and dataset_name == 'oasst1'):
        dataset = dataset.map(lambda x: {
            'input': '',
            'output': x['text'],
        })
    elif dataset_format == 'wikitext2' or (dataset_format is None and dataset_name == 'wikitext2'):
        dataset = dataset.map(lambda x: {
            'input': '',
            'output': x['text'],
        })
    elif dataset_format == 'input-output':
        # leave as is
        pass
    elif dataset_format == None:
        pass

    # Remove unused columns.
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
    )
    return dataset

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args, calibration=False) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples
        - mmlu-pro; gpqa; wikitext; mathqa

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    if not calibration:
         # Load dataset.
        dataset = load_data(args.dataset_name)
        dataset = format_dataset(dataset, args.dataset_format, args.dataset_name)
        
        # filer too long text
        def filter_max_len(text, max_len):
            length = len(tokenizer.tokenize(text))
            return length <= max_len

        dataset = dataset.filter(lambda x: filter_max_len(x["input"], args.source_max_len) and filter_max_len(x["output"], args.target_max_len) )
        # dataset = dataset.filter(lambda x: len(x["input"]) <= args.source_max_len and len(x["output"]) <= args.target_max_len )

        # Split train/eval, reduce size
        if args.do_predict or args.do_eval:
            if 'test' in dataset:
                # if "train" not in dataset :
                #     dataset = dataset["test"].train_test_split(
                #     test_size=args.eval_dataset_size, shuffle=True, seed=args.seed)
                pass
            elif "train" in dataset:
                print('Splitting train dataset in train and validation according to `eval_dataset_size`')
                dataset = dataset["train"].train_test_split(
                    test_size=args.eval_dataset_size, shuffle=True, seed=args.seed
                )
                
            eval_dataset = dataset['test']

            if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
                idx = random.sample(range(len(eval_dataset)), args.max_eval_samples)
                eval_dataset = eval_dataset.select(idx)
            if args.group_by_length:
                eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
        
        if args.do_train:
            if "train" not in dataset and "test" in dataset:
                dataset = dataset["test"].train_test_split(
                    test_size=args.eval_dataset_size, shuffle=True, seed=args.seed)
            
            train_dataset = dataset['train']
            if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
                idx = random.sample(range(len(train_dataset)), args.max_train_samples)
                train_dataset = train_dataset.select(idx)
            if args.group_by_length:
                train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            train_on_source=args.train_on_source,
            predict_with_generate=args.predict_with_generate,
        )
        return dict(
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            predict_dataset=eval_dataset if args.do_predict else None,
            data_collator=data_collator
        )

    else:
        dataset = load_data(args.cal_dataset_name)
        dataset = format_dataset(dataset, args.cal_dataset_format, args.cal_dataset_name)

        def filter_max_len(text, max_len):
            length = len(tokenizer.tokenize(text))
            return length <= max_len

        dataset = dataset.filter(lambda x: filter_max_len(x["input"], args.source_max_len) and filter_max_len(x["output"], args.target_max_len) )
        # dataset = dataset.filter(lambda x: len(x["input"]) <= args.source_max_len and len(x["output"]) <= args.target_max_len )

        cal_dataset = dataset['train'] # use training dataset as cal_dataset
        if args.max_cal_samples is not None and len(cal_dataset) > args.max_cal_samples:
            idx = random.sample(range(len(cal_dataset)), args.max_cal_samples)
            cal_dataset = cal_dataset.select(idx)
        if args.group_by_length:
            cal_dataset = cal_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            train_on_source=args.train_on_source,
            predict_with_generate=args.predict_with_generate,
        )
        return dict(
            eval_dataset=cal_dataset,
            data_collator=data_collator
        )
