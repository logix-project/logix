import gc
import os
import random
import struct
from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          default_data_collator)
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
    def __init__(self) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            "gpt2",
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            from_tf=False,
            config=self.config,
            ignore_mismatched_sizes=False,
            trust_remote_code=True,
        )
        replace_conv1d_modules(self.model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


def construct_model() -> nn.Module:
    return LanguageModel()


def get_loaders(
    eval_batch_size: int = 16,
    train_indices: Optional[List[int]] = None,
    valid_indices: Optional[List[int]] = None,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    train_batch_size = 8

    train_loader = get_wiki_dataloader(
        batch_size=train_batch_size,
        split="train",
        indices=train_indices,
    )
    eval_train_loader = get_wiki_dataloader(
        batch_size=eval_batch_size,
        split="eval_train",
        indices=train_indices,
    )
    valid_loader = get_wiki_dataloader(
        batch_size=eval_batch_size,
        split="valid",
        indices=valid_indices,
    )
    return train_loader, eval_train_loader, valid_loader


def get_wiki_dataloader(
    batch_size: int,
    split: str = "train",
    indices: List[int] = None,
) -> torch.utils.data.DataLoader:
    import ipdb; ipdb.set_trace(context=10)
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2", use_fast=True, trust_remote_code=True
    )

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    block_size = 512

    def group_texts(examples):
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    if split == "train" or split == "eval_train":
        train_dataset = lm_datasets["train"]
        ds = train_dataset
    else:
        eval_dataset = lm_datasets["validation"]
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=split == "train",
        collate_fn=default_data_collator,
    )


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def reset_seed() -> None:
    """Reset the seed to have randomized experiments."""
    rng_seed = struct.unpack("I", os.urandom(4))[0]
    set_seed(rng_seed)


def clear_gpu_cache() -> None:
    """Perform garbage collection and empty GPU cache reserved by Pytorch."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
