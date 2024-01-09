import torch
import torchvision
import torch.nn as nn
import numpy as np


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
