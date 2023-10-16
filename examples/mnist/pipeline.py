from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision


def construct_mlp(num_inputs: int = 784, num_classes: int = 10) -> nn.Module:
    model = torch.nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, 512, bias=True),
        nn.ReLU(),
        nn.Linear(512, 512, bias=True),
        nn.ReLU(),
        nn.Linear(512, 512, bias=True),
        nn.ReLU(),
        nn.Linear(512, num_classes, bias=True),
    )
    return model


def get_hyperparameters(data_name: str) -> Dict[str, float]:
    if data_name == "mnist":
        lr = 0.03
        wd = 0.0001
    elif data_name == "fmnist":
        lr = 0.01
        wd = 0.0001
    else:
        raise NotImplementedError()
    return {"lr": lr, "wd": wd}


def get_loaders(
    data_name: str,
    eval_batch_size: int = 2048,
    train_indices: Optional[List[int]] = None,
    valid_indices: Optional[List[int]] = None,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    assert data_name in ["mnist", "fmnist"]
    train_batch_size = 512

    if data_name == "mnist":
        train_loader = get_mnist_dataloader(
            batch_size=train_batch_size,
            split="train",
            indices=train_indices,
        )
        eval_train_loader = get_mnist_dataloader(
            batch_size=eval_batch_size,
            split="eval_train",
            indices=train_indices,
        )
        valid_loader = get_mnist_dataloader(
            batch_size=eval_batch_size,
            split="valid",
            indices=valid_indices,
        )
    else:
        train_loader = get_fmnist_dataloader(
            batch_size=train_batch_size,
            split="train",
            indices=train_indices,
        )
        eval_train_loader = get_fmnist_dataloader(
            batch_size=eval_batch_size,
            split="eval_train",
            indices=train_indices,
        )
        valid_loader = get_fmnist_dataloader(
            batch_size=eval_batch_size,
            split="valid",
            indices=valid_indices,
        )
    return train_loader, eval_train_loader, valid_loader


def get_mnist_dataloader(
    batch_size: int = 128,
    split: str = "train",
    indices: List[int] = None,
) -> torch.utils.data.DataLoader:
    assert split in ["train", "eval_train", "valid"]

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = torchvision.datasets.MNIST(
        root="/tmp/mnist/",
        download=True,
        train=split in ["train", "eval_train"],
        transform=transforms,
    )

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=split == "train",
        batch_size=batch_size,
        num_workers=0,
        drop_last=split == "train",
        pin_memory=True,
    )


def get_fmnist_dataloader(
    batch_size: int = 128,
    split: str = "train",
    indices: List[int] = None,
) -> torch.utils.data.DataLoader:
    assert split in ["train", "eval_train", "valid"]

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    dataset = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist/",
        download=True,
        train=split in ["train", "eval_train"],
        transform=transforms,
    )

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=split == "train",
        batch_size=batch_size,
        num_workers=0,
        drop_last=split == "train",
        pin_memory=True,
    )
