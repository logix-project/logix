import argparse
from types import SimpleNamespace
from tqdm import tqdm

import torch
from accelerate import Accelerator
import analog

from utils import (
    prepare_dataloader,
    prepare_model_optimizer_scheduler,
    Accuracy,
    save_model,
    loss_fn,
)


@torch.no_grad()
def validate(model, eval_dataloader):
    model.eval()
    eval_acc = Accuracy()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            loss = loss_fn(out.logits, batch["labels"])
        eval_acc.update(out.logits, batch["labels"])
    print(f"eval_loss: {loss.item()}, eval_accuracy: {eval_acc.compute()}")
    model.train()

    return eval_acc.compute()


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
    model, optimizer, scheduler = prepare_model_optimizer_scheduler(config)

    accelerator = Accelerator(mixed_precision=config.precision)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    acc = Accuracy()
    best_acc = -1
    model.train()
    for epoch in range(config.epochs):
        for step, batch in enumerate(tqdm(train_dataloader)):
            out = model(**batch)
            loss = (
                loss_fn(out.logits, batch["labels"])
                / config.gradient_accumulation_steps
            )
            loss.backward()

            if step % (config.log_steps * config.gradient_accumulation_steps) == 0:
                log_loss = loss.item() * config.gradient_accumulation_steps
                log_acc = acc.update(out.logits, batch["labels"])
                print(f"[Step {step}] train_loss: {log_loss} || train_acc: {log_acc}")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # Validation
        val_acc = validate(model, eval_dataloader)
        if val_acc > best_acc:
            save_model(
                model,
                model_name=config.model_id.replace("/", "_"),
                models_folder="files/",
            )


if __name__ == "__main__":
    main()
