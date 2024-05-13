import argparse

import logix
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm

from utils import get_loader, get_model, get_tokenizer, set_seed

# Enable TF32 if possible
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True


def main():
    parser = argparse.ArgumentParser("GPT2 Influence Analysis")
    parser.add_argument("--config_path", type=str, default="./config.yaml")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hessian", type=str, default="kfac")
    parser.add_argument("--lora", type=str, default="random")
    parser.add_argument("--save", type=str, default="grad")
    parser.add_argument("--mlp_only", action="store_true")
    parser.add_argument("--data_name", type=str, default="openwebtext")
    args = parser.parse_args()

    set_seed(0)
    accelerator = Accelerator()

    # prepare model & data loader
    model = get_model(model_name=args.model_name, cache_dir=args.cache_dir)
    tokenizer = get_tokenizer(
        model_name=args.model_name, cache_dir=args.cache_dir
    )
    data_loader = get_loader(
        model_name=args.model_name,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        data_name=args.data_name,
    )

    # LogIX Setup
    model_name_strip = args.model_name.split("/")[-1]
    project = f"{model_name_strip}_{args.lora}_{args.hessian}_{args.data_name}"
    name_filter = ["att", "mlp"]
    if args.mlp_only:
        project += "_mlp"
        name_filter = ["mlp"]

    run = logix.init(project, config=args.config_path)
    run.watch(model, name_filter=name_filter)
    if args.lora != "none":
        run.add_lora()
    scheduler = logix.LogIXScheduler(
        run, lora="none", hessian=args.hessian, save=args.save
    )

    # Extract log
    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()
    for _ in scheduler:
        for batch in tqdm(data_loader):
            data_id = tokenizer.batch_decode(batch["input_ids"])
            targets = batch.pop("labels")
            with run(data_id=data_id, mask=batch["attention_mask"]):
                model.zero_grad()
                lm_logits = model(**batch)
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = targets[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                    ignore_index=-100,
                )
                accelerator.backward(loss)
        logix.finalize()
    print(f"Log saved in {project}")


if __name__ == "__main__":
    main()
