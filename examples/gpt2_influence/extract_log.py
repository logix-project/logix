import argparse

from tqdm import tqdm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
import logix
from logix.statistic import Covariance

from utils import construct_model, get_loaders, set_seed


def main():
    parser = argparse.ArgumentParser("GPT2 Influence Analysis")
    parser.add_argument("--project", type=str, default="wiki")
    parser.add_argument("--config_path", type=str, default="./config.yaml")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hessian", type=str, default="kfac")
    parser.add_argument("--lora", type=str, default="random")
    parser.add_argument("--save", type=str, default="grad")
    args = parser.parse_args()

    set_seed(0)
    accelerator = Accelerator()

    # prepare model & data loader
    model, tokenizer = construct_model(resume=True)
    model.eval()
    train_loader = get_loaders(eval_batch_size=args.batch_size)[1]
    model, train_loader = accelerator.prepare(model, train_loader)

    # Set-up
    run = logix.init(args.project, config=args.config_path)
    run.watch(model, name_filter=["attn", "mlp"])
    scheduler = logix.LogIXScheduler(
        run, lora=args.lora, hessian=args.hessian, save=args.save
    )
    for _ in scheduler:
        for batch in tqdm(train_loader):
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


if __name__ == "__main__":
    main()
