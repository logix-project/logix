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

import argparse

from gpt_utils import construct_model, get_datasets, set_seed
from transformers import Trainer, TrainingArguments, default_data_collator

from logix.huggingface import LogIXArguments, patch_trainer


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
    train_dataset = get_datasets()[0]

    logix_args = LogIXArguments(
        project=args.project,
        config=args.config_path,
        lora=True,
        hessian="raw",
        save="grad",
        label_key="input_ids",
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
    trainer.extract_log()


if __name__ == "__main__":
    main()
