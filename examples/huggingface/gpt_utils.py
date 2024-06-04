# Copyright 2023-present the LogIX team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import struct
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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


def construct_model(resume=False):
    config = AutoConfig.from_pretrained(
        "gpt2",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )
    replace_conv1d_modules(model)
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2", use_fast=True, trust_remote_code=True
    )
    if resume:
        state_dict = torch.load(
            "files/checkpoints/0/wiki_epoch_3.pt", map_location="cpu"
        )
        for key in list(state_dict.keys()):
            if "model." in key:
                state_dict[key.replace("model.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict)
    return model, tokenizer


def get_datasets():
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
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
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

    return lm_datasets["train"], lm_datasets["validation"]


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
