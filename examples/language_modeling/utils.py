import gc
import os
import random
import struct
from itertools import chain
import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)
from transformers.pytorch_utils import Conv1D


def replace_conv1d_modules(model):
    # GPT-2 is defined in terms of Conv1D. However, this does not work for EK-FAC.
    # Here, we convert these Conv1D modules to linear modules recursively.
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_conv1d_modules(module)

        if isinstance(module, Conv1D):
            new_module = nn.Linear(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
            )
            new_module.weight.data.copy_(module.weight.data.t())
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)


class LanguageModel(nn.Module):
    def __init__(self, model_name, cache_dir=None) -> None:
        super().__init__()
        if "gpt2" in model_name:
            self.config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                from_tf=False,
                config=self.config,
                ignore_mismatched_sizes=False,
                trust_remote_code=True,
            )
            replace_conv1d_modules(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=cache_dir
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


def get_model(model_name, cache_dir) -> nn.Module:
    return LanguageModel(model_name, cache_dir)


def get_tokenizer(
    model_name, cache_dir, add_padding_token=False
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=("gpt2" in model_name),
        cache_dir=cache_dir,
    )

    if tokenizer.pad_token is None and add_padding_token:
        print("No pad token found. Setting `<pad>` as a pad token.")
        tokenizer.pad_token = "<pad>"
        if "<pad>" not in tokenizer.get_vocab():
            tokenizer.add_tokens("<pad>")
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    return tokenizer


def get_dataset(
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    data_path: str,
    data_name: Optional[str] = None,
    split: str = "train",
    sample_ratio: float = 0.005,
    cache_dir: str = None,
) -> torch.utils.data.DataLoader:
    model_name_strip = model_name.split("/")[-1]
    save_data_name = data_path if data_name is None else data_name
    save_data_name = save_data_name.split("/")[-1]
    if os.path.exists(
        os.path.join(cache_dir, f"{model_name_strip}_{save_data_name}.pt")
    ):
        print("[*] Loading from cached data...")
        lm_datasets = load_from_disk(
            os.path.join(cache_dir, f"{model_name_strip}_{save_data_name}.pt")
        )
        return lm_datasets[split]

    # Prepare raw dataset
    if data_path == "external":
        data_kwargs = {
            "path": "json",
            "data_files": "./custom_data/external/data.json",
            "cache_dir": cachd_dir,
            "num_proc": 4,
        }
    elif data_path == "generated":
        data_kwargs = {
            "path": "json",
            "data_files": f"./custom_data/generated/{model_name_strip}/data.json",
            "cache_dir": cache_dir,
            "num_proc": 4,
        }
    else:
        data_kwargs = {
            "path": data_path,
            "name": data_name,
            "cache_dir": cache_dir,
            "num_proc": 4,
        }
    raw_datasets = load_dataset(**data_kwargs)

    if sample_ratio is not None:
        sampled_train = raw_datasets["train"].train_test_split(
            test_size=0.005, shuffle=True, seed=42
        )
        raw_datasets["train"] = sampled_train["test"]

    # Tokenize dataset
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    # Group text
    if data_path not in ["generated", "external"]:
        block_size = 512

        def group_texts(examples):
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = copy.deepcopy(result["input_ids"])
            return result

    else:
        block_size = "samples"

        def group_texts(examples):
            examples["labels"] = copy.deepcopy(examples["input_ids"])
            for label in examples["labels"]:
                for idx, token in enumerate(label):
                    if token == tokenizer.pad_token_id:
                        label[idx] = -100
            return examples

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    print("[*] Saving data to disk...")
    lm_datasets.save_to_disk(
        os.path.join(cache_dir, f"{model_name_strip}_{save_data_name}.pt")
    )

    return lm_datasets[split]


def get_loader(
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    data_path: str,
    data_name: Optional[str] = None,
    split: str = "train",
    cache_dir: str = None,
) -> torch.utils.data.DataLoader:
    dataset = get_dataset(
        model_name=model_name,
        tokenizer=tokenizer,
        data_path=data_path,
        data_name=data_name,
        split=split,
        cache_dir=cache_dir,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator
    )
    return dataloader


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
