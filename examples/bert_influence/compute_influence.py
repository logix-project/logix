import argparse

import torch
import torch.nn.functional as F
from accelerate import Accelerator
import logix
from logix.analysis import InfluenceFunction

from utils import construct_model, get_loaders


def main():
    parser = argparse.ArgumentParser("GLUE Influence Analysis")
    parser.add_argument("--project", type=str, default="sst2")
    parser.add_argument("--config_path", type=str, default="./config.yaml")
    parser.add_argument("--data_name", type=str, default="sst2")
    parser.add_argument("--damping", type=float, default=None)
    args = parser.parse_args()

    # prepare model & data loader
    model, tokenizer = construct_model(
        args.data_name, ckpt_path=f"files/checkpoints/0/{args.data_name}_epoch_3.pt"
    )
    model.eval()
    test_loader = get_loaders(data_name=args.data_name)[-1]

    accelerator = Accelerator()
    model, test_loader = accelerator.prepare(model, test_loader)

    # Set-up LogIX
    run = logix.init(args.project, config=args.config_path)

    logix.watch(model)
    logix.initialize_from_log()
    log_loader = logix.build_log_dataloader()

    # influence analysis
    logix.setup({"log": "grad"})
    logix.eval()
    for batch in test_loader:
        data_id = tokenizer.batch_decode(batch["input_ids"])
        labels = batch.pop("labels").view(-1)
        _ = batch.pop("idx")
        with run(data_id=data_id, mask=batch["attention_mask"]):
            model.zero_grad()
            outputs = model(**batch)
            logits = outputs.view(-1, outputs.shape[-1])
            loss = F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)
            accelerator.backward(loss)

        test_log = run.get_log()
        run.influence.compute_influence_all(test_log, log_loader, damping=args.damping)
        break


if __name__ == "__main__":
    main()
