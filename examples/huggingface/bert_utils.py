from typing import List, Optional, Tuple, Union

import gc
import os
import random
import struct

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
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


def construct_model(data_name: str, ckpt_path: Union[None, str] = None) -> nn.Module:
    config = AutoConfig.from_pretrained(
        "bert-base-cased",
        num_labels=2,
        finetuning_task=data_name,
        trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased", use_fast=True, trust_remote_code=True
    )
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        for key in list(state_dict.keys()):
            if "model." in key:
                state_dict[key.replace("model.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {ckpt_path}.")
    return model, tokenizer


def get_datasets(
    data_name: str,
    train_indices: Optional[List[int]] = None,
    valid_indices: Optional[List[int]] = None,
) -> Tuple[
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
]:
    assert data_name in ["qnli", "sst2"]
    train_dataset = get_dataset(
        data_name=data_name, split="train", indices=train_indices
    )
    eval_train_dataset = get_dataset(
        data_name=data_name, split="eval_train", indices=train_indices
    )
    valid_dataset = get_dataset(
        data_name=data_name, split="valid", indices=valid_indices
    )
    return train_dataset, eval_train_dataset, valid_dataset


def get_dataset(
    data_name: str,
    batch_size: int = 32,
    split: str = "train",
    indices: List[int] = None,
    do_not_pad: bool = False,
) -> torch.utils.data.Dataset:
    assert data_name in ["qnli", "sst2"]

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

    return ds
