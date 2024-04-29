import math
from typing import Dict, List, Optional, Tuple

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


class SequenceClassificationModel(nn.Module):
    def __init__(self, data_name: str) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            "roberta-base",
            num_labels=2,
            finetuning_task=data_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
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


def construct_model(data_name: str) -> nn.Module:
    return SequenceClassificationModel(data_name=data_name)


def get_hyperparameters(data_name: str) -> Dict[str, float]:
    assert data_name in ["sst2", "qnli", "rte"]
    if data_name == "sst2":
        lr = 3e-05
        wd = 0.01
    elif data_name == "qnli":
        lr = 1e-05
        wd = 0.01
    elif data_name == "rte":
        lr = 2e-05
        wd = 0.01
    else:
        raise NotImplementedError()
    return {"lr": lr, "wd": wd, "epochs": 4}


def get_loaders(
    data_name: str,
    eval_batch_size: int = 32,
    train_indices: Optional[List[int]] = None,
    valid_indices: Optional[List[int]] = None,
    do_corrupt: bool = False,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    assert data_name in ["qnli", "sst2", "rte"]
    train_batch_size = 32

    train_loader = get_dataloader(
        data_name=data_name,
        batch_size=train_batch_size,
        split="train",
        indices=train_indices,
        do_corrupt=do_corrupt,
    )
    eval_train_loader = get_dataloader(
        data_name=data_name,
        batch_size=eval_batch_size,
        split="eval_train",
        indices=train_indices,
        do_corrupt=do_corrupt,
    )
    valid_loader = get_dataloader(
        data_name=data_name,
        batch_size=eval_batch_size,
        split="valid",
        indices=valid_indices,
        do_corrupt=False,
    )
    return train_loader, eval_train_loader, valid_loader


def get_dataloader(
    data_name: str,
    batch_size: int = 32,
    split: str = "train",
    indices: List[int] = None,
    do_corrupt: bool = False,
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
        "roberta-base", use_fast=True, trust_remote_code=True
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
            *texts, padding=padding, max_length=max_seq_length, truncation=True, return_token_type_ids=True
        )
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=(not False),
    )

    if split == "train" or split == "eval_train":
        train_dataset = raw_datasets["train"]
        ds = train_dataset
        if data_name != "rte":
            ds = ds.select(range(51_200))
        else:
            ds = ds.select(range(2432))
    else:
        eval_dataset = raw_datasets["validation"]
        ds = eval_dataset
        # if data_name != "rte":
        #     ds = ds.select(range(512))

    if do_corrupt:
        if split == "valid":
            raise NotImplementedError(
                "Performing corruption on the validation dataset is not supported."
            )

        num_corrupt = math.ceil(len(ds) * 0.1)

        def corrupt_function(example, idx):
            if idx < num_corrupt:
                # Switch the labels.
                example["labels"] = (example["labels"] + 1) % 2
            return example

        ds = ds.map(
            corrupt_function,
            with_indices=True,
            batched=False,
            load_from_cache_file=(not False),
        )

    if indices is not None:
        ds = ds.select(indices)

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=split == "train",
        collate_fn=default_data_collator,
        num_workers=4,
    )
