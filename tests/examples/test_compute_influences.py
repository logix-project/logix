import unittest
import torch
import torchvision
import torch.nn as nn
import numpy as np
import os

DEVICE = torch.device("cpu")


def construct_mlp(num_inputs=784, num_classes=10):
    return torch.nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, 4, bias=False),
        nn.ReLU(),
        nn.Linear(4, 2, bias=False),
        nn.ReLU(),
        nn.Linear(2, num_classes, bias=False),
    )


def get_mnist_dataloader(
    batch_size=128,
    split="train",
    shuffle=False,
    subsample=False,
    indices=None,
    drop_last=False,
):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    is_train = split == "train"
    dataset = torchvision.datasets.MNIST(
        root="/tmp/mnist/", download=True, train=is_train, transform=transforms
    )

    if subsample and split == "train" and indices is None:
        dataset = torch.utils.data.Subset(dataset, np.arange(6_000))

    if indices is not None:
        if subsample and split == "train":
            print("Overriding `subsample` argument as `indices` was provided.")
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=0,
        drop_last=drop_last,
    )


class TestSingleCheckpointInfluence(unittest.TestCase):
    def test_single_checkpoint_influence(self):
        data_name = "mnist"
        eval_idxs = (0,)

        model = construct_mlp().to(DEVICE)
        # Get a single checkpoint (first model_id and last epoch).
        model.load_state_dict(
            torch.load(
                f"{os.path.dirname(os.path.abspath(__file__))}/checkpoints/mnist_0_epoch_9.pt",
                map_location="cpu",
            )
        )
        model.eval()
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        dataloader_fn = get_mnist_dataloader
        train_loader = dataloader_fn(
            batch_size=512, split="train", shuffle=False, subsample=True
        )
        query_loader = dataloader_fn(
            batch_size=1, split="valid", shuffle=False, indices=eval_idxs
        )

        # Set-up
        from analog import AnaLog
        from analog.utils import DataIDGenerator

        analog = AnaLog(project="test", config="examples/mnist/config.yaml")

        # Gradient & Hessian logging
        analog.watch(model, name_filter=["1", "3", "5"])
        id_gen = DataIDGenerator()
        for inputs, targets in train_loader:
            data_id = id_gen(inputs)
            analog.setup({"log": "grad", "save": "grad", "statistic": "kfac"})
            with analog(data_id=data_id):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                model.zero_grad()
                outs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
                loss.backward()
        analog.finalize()

        # Influence Analysis
        log_loader = analog.build_log_dataloader()

        from analog.analysis import InfluenceFunction

        analog.add_analysis({"influence": InfluenceFunction})
        query_iter = iter(query_loader)
        analog.eval()
        with analog(data_id=["test"]) as al:
            test_input, test_target = next(query_iter)
            test_input, test_target = test_input.to(DEVICE), test_target.to(DEVICE)
            model.zero_grad()
            test_out = model(test_input)
            test_loss = torch.nn.functional.cross_entropy(
                test_out, test_target, reduction="sum"
            )
            test_loss.backward()
            test_log = al.get_log()
        analog.influence.compute_influence_all(test_log, log_loader)
