import argparse
from types import SimpleNamespace
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from accelerate import Accelerator
import logix

from utils import (
    prepare_dataloader,
    prepare_model_optimizer_scheduler,
    Accuracy,
    loss_fn,
)


def main():
    parser = argparse.ArgumentParser("llama log extraction")
    parser.add_argument("--model_id", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--dataset_name", type=str, default="alpaca-gpt4")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--n_freeze", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n_eval_samples", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--log_model", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--freeze_embed", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42)

    config = parser.parse_args()

    train_dataloader, eval_dataloader = prepare_dataloader(config)
    config.total_train_steps = (
        config.epochs * len(train_dataloader) // config.gradient_accumulation_steps
    )
    # TODO: NEED TO IMPLEMENT MODEL LOADING LOGIC
    model, _, _ = prepare_model_optimizer_scheduler(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    accelerator = Accelerator(mixed_precision=config.precision)
    model, train_dataloader = accelerator.prepare(model, train_dataloader)

    run = logix.init(project="llama_test", config="config.yaml")
    logix.watch(model, name_filter=["layers"])
    logix.add_lora()
    scheduler = logix.LogiXScheduler(run, ekfac=True)

    model.train()
    for epoch in scheduler:
        for step, batch in enumerate(tqdm(train_dataloader)):
            data_id = tokenizer.batch_decode(batch["input_ids"])
            with run(data_id=data_id):
                model.zero_grad()
                out = model(**batch)
                loss = loss_fn(out.logits, batch["labels"], reduction="sum")
                accelerator.backward(loss)
        logix.finalize()


if __name__ == "__main__":
    main()
