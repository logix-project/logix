import os
import copy
import argparse

import torch
import torch.nn.functional as F
from accelerate import Accelerator
import logix
from logix.analysis import InfluenceFunction
from logix.utils import merge_logs
from tqdm import tqdm

from utils import get_model, get_tokenizer, get_loader, set_seed


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
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--hessian", type=str, default="kfac")
    parser.add_argument("--lora", type=str, default="random")
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--mlp_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--damping", type=float, default=1e-5)
    args = parser.parse_args()

    set_seed(0)
    accelerator = Accelerator()

    # prepare model & data loader
    model = get_model(model_name=args.model_name, cache_dir=args.cache_dir)
    tokenizer = get_tokenizer(model_name=args.model_name, cache_dir=args.cache_dir)
    data_loader = get_loader(
        model_name=args.model_name,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
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
    logix.setup({"log": "grad"})
    logix.eval()
    if_scores = []
    train_ids = None
    test_ids = []
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

        if idx == 7 or idx == len(data_loader) - 1:
            merged_test_log = merge_logs(merged_test_logs)
            if_score, train_ids_batch = run.influence.compute_influence_all(
                merged_test_log, log_loader
            )
            if_scores.append(if_score)
            if train_ids is None:
                train_ids = train_ids_batch
            else:
                assert train_ids == train_ids_batch
            test_ids.extend(merged_test_log[0])
            if_scores = torch.cat(if_scores, dim=0)
            merged_test_logs = []
            break
    base_dir = "/data/tir/projects/tir3/users/sangkeuc/gpt/results/"
    save_dir = os.path.join(
        base_dir, f"{args.split}_{model_name_strip}_{args.lora}_{args.hessian}"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(if_scores, os.path.join(save_dir, "scores.pt"))
    torch.save(train_ids, os.path.join(save_dir, "train_ids.pt"))
    torch.save(test_ids, os.path.join(save_dir, "test_ids.pt"))


if __name__ == "__main__":
    main()
