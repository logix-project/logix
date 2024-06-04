import gc
import os
import random
import struct
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)

GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


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


class SequenceClassificationModel(nn.Module):
    def __init__(self, data_name: str) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            "bert-base-cased",
            num_labels=2,
            finetuning_task=data_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased",
            from_tf=False,
            config=self.config,
            ignore_mismatched_sizes=False,
            trust_remote_code=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ).logits


def construct_model(data_name: str, ckpt_path: Union[None, str] = None) -> nn.Module:
    model = SequenceClassificationModel(data_name)
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased", use_fast=True, trust_remote_code=True
    )
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"Loaded model from {ckpt_path}.")
    return model, tokenizer


def get_loaders(
    data_name: str,
    eval_batch_size: int = 32,
    train_indices: Optional[List[int]] = None,
    valid_indices: Optional[List[int]] = None,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    assert data_name in ["qnli", "sst2", "rte"]
    train_batch_size = 16

    train_loader = get_dataloader(
        data_name=data_name,
        batch_size=train_batch_size,
        split="train",
        indices=train_indices,
    )
    eval_train_loader = get_dataloader(
        data_name=data_name,
        batch_size=eval_batch_size,
        split="eval_train",
        indices=train_indices,
    )
    valid_loader = get_dataloader(
        data_name=data_name,
        batch_size=eval_batch_size,
        split="valid",
        indices=valid_indices,
    )
    return train_loader, eval_train_loader, valid_loader


def get_dataloader(
    data_name: str,
    batch_size: int = 32,
    split: str = "train",
    indices: List[int] = None,
    do_not_pad: bool = False,
) -> torch.utils.data.DataLoader:
    assert data_name in ["qnli", "sst2", "rte"]

    raw_datasets = load_dataset(
        "glue",
        data_name,
    )
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
    assert num_labels == 2

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased", use_fast=True, trust_remote_code=True
    )

    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[data_name]
    padding = "max_length"
    max_seq_length = 128
    if do_not_pad:
        padding = "do_not_pad"

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts, padding=padding, max_length=max_seq_length, truncation=True
        )
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=(not False),
        desc="Running tokenizer on dataset",
    )

    if split == "train" or split == "eval_train":
        train_dataset = raw_datasets["train"]
        ds = train_dataset
    else:
        eval_dataset = raw_datasets["validation"]
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=split == "train",
        collate_fn=default_data_collator,
    )
