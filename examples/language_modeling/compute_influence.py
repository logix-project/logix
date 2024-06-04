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
import copy
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from utils import get_loader, get_model, get_tokenizer, set_seed

import logix
from logix.utils import merge_logs

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True


def main():
    parser = argparse.ArgumentParser("GPT2 Influence Analysis")
    parser.add_argument("--config_path", type=str, default="./config.yaml")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data/tir/projects/tir3/users/sangkeuc/huggingface",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data/tir/projects/tir3/users/sangkeuc/gpt/results",
    )
    parser.add_argument("--model_name", type=str, default="gpt2-xl")
    parser.add_argument("--data_path", type=str, default="wikitext")
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hessian", type=str, default="raw")
    parser.add_argument("--lora", type=str, default="random")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--mlp_only", action="store_true")
    parser.add_argument("--damping", type=float, default=1e-5)
    args = parser.parse_args()

    set_seed(0)
    accelerator = Accelerator()

    # prepare model, tokenizer, data loader
    model = get_model(model_name=args.model_name, cache_dir=args.cache_dir)
    tokenizer = get_tokenizer(
        model_name=args.model_name, cache_dir=args.cache_dir, add_padding_token=True
    )
    data_loader = get_loader(
        model_name=args.model_name,
        data_path=args.data_path,
        data_name=args.data_name,
        tokenizer=tokenizer,
        batch_size=1,
        cache_dir=args.cache_dir,
        split=args.split,
    )
    model, data_loader = accelerator.prepare(model, data_loader)

    # Set-up LogIX
    model_name_strip = args.model_name.split("/")[-1]
    project = f"{model_name_strip}_{args.lora}_{args.hessian}"
    name_filter = ["att", "mlp"]
    if args.mlp_only:
        project += "_mlp"
        name_filter = ["mlp"]

    run = logix.init(project, config=args.config_path)
    logix.watch(model, name_filter=name_filter)
    logix.initialize_from_log()
    log_loader = logix.build_log_dataloader(batch_size=64)

    # Influence analysis
    logix.setup({"grad": ["log"]})
    logix.eval()
    merged_test_logs = []
    for idx, batch in enumerate(tqdm(data_loader)):
        data_id = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        targets = batch.pop("labels")
        with run(data_id=data_id, mask=batch["attention_mask"]):
            model.zero_grad()
            logits = model(**batch)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
                ignore_index=-100,
            )
            accelerator.backward(loss)

        test_log = logix.get_log()
        merged_test_logs.append(copy.deepcopy(test_log))

        if idx == args.batch_size - 1:
            break

    merged_test_log = merge_logs(merged_test_logs)
    result = run.influence.compute_influence_all(merged_test_log, log_loader)

    save_dir = "./save"
    torch.save(result["influence"], os.path.join(save_dir, "scores.pt"))
    torch.save(result["src_ids"], os.path.join(save_dir, "test_ids.pt"))
    torch.save(result["tgt_ids"], os.path.join(save_dir, "train_ids.pt"))


if __name__ == "__main__":
    main()
