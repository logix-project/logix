import time
import argparse

import torch
from accelerate import Accelerator
from analog import AnaLog
from analog.utils import DataIDGenerator, get_world_size, get_rank
from analog.analysis import InfluenceFunction

from train import (
    get_mnist_dataloader,
    get_fmnist_dataloader,
    construct_mlp,
)


def main():
    parser = argparse.ArgumentParser("MNIST Influence Analysis")
    parser.add_argument("--data", type=str, default="mnist", help="mnist or fmnist")
    parser.add_argument("--eval-idxs", type=int, nargs="+", default=[0])
    parser.add_argument("--damping", type=float, default=1e-5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    accelerator = Accelerator()

    model = construct_mlp()
    model.load_state_dict(
        torch.load(f"checkpoints/{args.data}_0_epoch_9.pt", map_location="cpu")
    )
    model.eval()

    dataloader_fn = (
        get_mnist_dataloader if args.data == "mnist" else get_fmnist_dataloader
    )
    train_loader = dataloader_fn(
        batch_size=500, split="train", shuffle=False, subsample=True
    )
    query_loader = dataloader_fn(
        batch_size=1, split="valid", shuffle=False, indices=args.eval_idxs
    )

    model, train_loader, query_loader = accelerator.prepare(
        model, train_loader, query_loader
    )

    analog = AnaLog(project="test", config="config.yaml")

    analog.watch(model)
    analog.setup({"log": "grad", "statistic": "kfac"})
    id_gen = DataIDGenerator()

    if not args.resume:
        # Epoch 0: Compute Hessian
        # Epoch 1: Gradient logging w/ PCA+LoRA
        for epoch in range(2):
            for inputs, targets in train_loader:
                data_id = id_gen(inputs)
                with analog(data_id=data_id):
                    model.zero_grad()
                    outs = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        outs, targets, reduction="sum"
                    )
                    loss.backward()
            analog.finalize()
            if epoch == 0:
                analog.setup({"save": "grad", "log": "grad", "statistic": "kfac"})
                analog.add_lora()
    else:
        analog.add_lora()
        analog.initialize_from_log()
    print(analog.get_covariance_state())

    log_loader = analog.build_log_dataloader()
    analog.eval()

    analog.add_analysis({"influence": InfluenceFunction})
    query_iter = iter(query_loader)
    with analog(data_id=["test"]):
        test_input, test_target = next(query_iter)
        model.zero_grad()
        test_out = model(test_input)
        test_loss = torch.nn.functional.cross_entropy(
            test_out, test_target, reduction="sum"
        )
        test_loss.backward()
    test_log = analog.get_log()
    start = time.time()
    if_scores = analog.influence.compute_influence_all(
        test_log, log_loader, damping=args.damping
    )
    _, top_influential_data = torch.topk(if_scores, k=10)

    # Save
    if_scores = if_scores.numpy().tolist()
    torch.save(if_scores, "if_distributed.pt")[0]
    print("Computation time:", time.time() - start)
    print("Top influential data indices:", top_influential_data.numpy().tolist())


if __name__ == "__main__":
    main()
