import argparse

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from transformers import default_data_collator, Trainer, TrainingArguments

import logix
from logix.huggingface import patch_trainer, LogIXArguments
from gpt_utils import construct_model, get_datasets, set_seed


def main():
    parser = argparse.ArgumentParser("GLUE Influence Analysis")
    parser.add_argument("--project", type=str, default="wiki")
    parser.add_argument("--config_path", type=str, default="./config.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_name", type=str, default="sst2")
    args = parser.parse_args()

    set_seed(0)

    # prepare model & data loader
    model, tokenizer = construct_model(resume=True)
    model.eval()
    train_dataset = get_datasets()[-1]

    logix_args = LogIXArguments(
        project=args.project,
        config=args.config_path,
        lora=True,
        hessian="raw",
        save="grad",
        label_key="input_ids",
        initialize_from_log=True,
        log_batch_size=args.batch_size,
    )
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        report_to="none",
    )

    LogIXTrainer = patch_trainer(Trainer)
    trainer = LogIXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        args=training_args,
        logix_args=logix_args,
    )
    if_scores = trainer.influence()
    torch.save(if_scores, "gpt_influence.pt")


if __name__ == "__main__":
    main()
