import argparse

import torch
import torch.nn.functional as F
from accelerate import Accelerator
import analog
from analog.analysis import InfluenceFunction
from tqdm import tqdm

from utils import construct_model, get_loaders, set_seed


def main():
    parser = argparse.ArgumentParser("GPT2 Influence Analysis")
    parser.add_argument("--project", type=str, default="wiki")
    parser.add_argument("--config_path", type=str, default="./config.yaml")
    parser.add_argument("--damping", type=float, default=1e-5)
    args = parser.parse_args()

    set_seed(0)
    accelerator = Accelerator()

    # prepare model & data loader
    model, tokenizer = construct_model(resume=True)
    model.eval()
    test_loader = get_loaders(eval_batch_size=16)[-1]
    model, test_loader = accelerator.prepare(model, test_loader)

    # Set-up AnaLog
    run = analog.init(args.project, config=args.config_path)

    analog.watch(model, name_filter=["attn", "mlp"])
    analog.initialize_from_log()
    log_loader = analog.build_log_dataloader(batch_size=64)

    # Influence analysis
    analog.setup({"log": "grad"})
    analog.eval()
    if_scores = []
    for batch in test_loader:
        data_id = tokenizer.batch_decode(batch["input_ids"])
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

        test_log = analog.get_log()
        if_score = run.influence.compute_influence_all(test_log, log_loader)
        if_scores.append(if_score)
    if_scores = torch.cat(if_scores, dim=0)
    torch.save(if_scores, "scores.pt")


if __name__ == "__main__":
    main()
