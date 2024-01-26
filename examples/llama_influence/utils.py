from pathlib import Path
import json
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import default_data_collator
from transformers import get_cosine_schedule_with_warmup


def prompt_no_input(row):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ).format_map(row)


def prompt_input(row):
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ).format_map(row)


def create_alpaca_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)


def pad_eos(ds):
    EOS_TOKEN = "</s>"
    return [f"{row['output']}{EOS_TOKEN}" for row in ds]


def pack(dataset, tokenizer, max_seq_len=1024):
    tkds_ids = tokenizer([s["example"] for s in dataset])["input_ids"]

    all_token_ids = []
    for tokenized_input in tkds_ids:
        all_token_ids.extend(tokenized_input)  # + [tokenizer.eos_token_id])

    print(f"Total number of tokens: {len(all_token_ids)}")
    packed_ds = []
    for i in range(0, len(all_token_ids), max_seq_len + 1):
        input_ids = all_token_ids[i : i + max_seq_len + 1]
        if len(input_ids) == (max_seq_len + 1):
            packed_ds.append(
                {"input_ids": input_ids[:-1], "labels": input_ids[1:]}
            )  # this shift is not needed if using the model.loss
    return packed_ds


def prepare_dataloader(config):
    dataset_file = "alpaca_gpt4_data.json"
    with open(dataset_file, "r") as f:
        alpaca = json.load(f)

    seed = config.seed

    random.seed(seed)
    random.shuffle(alpaca)  # this could also be a parameter
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = alpaca[:-1000]
    eval_dataset = alpaca[-1000:]

    train_prompts = [create_alpaca_prompt(row) for row in train_dataset]
    eval_prompts = [create_alpaca_prompt(row) for row in eval_dataset]

    train_outputs = pad_eos(train_dataset)
    eval_outputs = pad_eos(eval_dataset)

    train_dataset = [
        {"prompt": s, "output": t, "example": s + t}
        for s, t in zip(train_prompts, train_outputs)
    ]
    eval_dataset = [
        {"prompt": s, "output": t, "example": s + t}
        for s, t in zip(eval_prompts, eval_outputs)
    ]

    train_ds_packed = pack(train_dataset, tokenizer, config.max_seq_len)
    eval_ds_packed = pack(eval_dataset, tokenizer, config.max_seq_len)

    train_dataloader = DataLoader(
        train_ds_packed,
        batch_size=config.batch_size,
        collate_fn=default_data_collator,  # we don't need any special collator 😎
        shuffle=config.shuffle,
    )

    eval_dataloader = DataLoader(
        eval_ds_packed,
        batch_size=config.batch_size,
        collate_fn=default_data_collator,
        shuffle=False,
    )

    return train_dataloader, eval_dataloader


def prepare_model_optimizer_scheduler(config):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map=0,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    for param in model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True
    for param in model.model.layers[config.n_freeze :].parameters():
        param.requires_grad = True

    if config.freeze_embed:
        model.model.embed_tokens.weight.requires_grad_(False)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    optim = torch.optim.Adam(
        model.parameters(), lr=config.lr, betas=(0.9, 0.99), eps=1e-5
    )
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_training_steps=config.total_train_steps,
        num_warmup_steps=config.total_train_steps // 10,
    )

    return model, optim, scheduler


def loss_fn(x, y, reduction="mean"):
    return torch.nn.functional.cross_entropy(
        x.view(-1, x.shape[-1]), y.view(-1), reduction=reduction
    )


def save_model(model, model_name, models_folder="models"):
    """Save the model to wandb as an artifact
    Args:
        model (nn.Module): Model to save.
        model_name (str): Name of the model.
        models_folder (str, optional): Folder to save the model. Defaults to "models".
    """
    file_name = Path(f"{models_folder}/{model_name}")
    file_name.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(file_name, safe_serialization=True)
    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    tokenizer.save_pretrained(model_name)


class Accuracy:
    "A simple Accuracy function compatible with HF models"

    def __init__(self):
        self.count = 0
        self.tp = 0.0

    def update(self, logits, labels):
        logits, labels = logits.argmax(dim=-1).view(-1).cpu(), labels.view(-1).cpu()
        tp = (logits == labels).sum()
        self.count += len(logits)
        self.tp += tp
        return tp / len(logits)

    def compute(self):
        return self.tp / self.count
