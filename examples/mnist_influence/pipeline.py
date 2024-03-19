import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision


def construct_model() -> nn.Module:
    model = torch.nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512, bias=True),
        nn.ReLU(),
        nn.Linear(512, 256, bias=True),
        nn.ReLU(),
        nn.Linear(256, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, 10, bias=True),
    )
    return model


def get_hyperparameters(data_name: str) -> Dict[str, float]:
    # See `scripts/find_hyperparameters.py` to examine how these values were selected.
    assert data_name in ["mnist", "fmnist"]
    del data_name
    return {"lr": 0.03, "wd": 0.001, "epochs": 20}


def get_loaders(
    data_name: str,
    eval_batch_size: int = 1024,
    train_indices: Optional[List[int]] = None,
    valid_indices: Optional[List[int]] = None,
    do_corrupt: bool = False,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    assert data_name in ["mnist", "fmnist"]
    train_batch_size = 64

    if data_name == "mnist":
        train_loader = get_mnist_dataloader(
            batch_size=train_batch_size,
            split="train",
            indices=train_indices,
            do_corrupt=do_corrupt,
        )
        eval_train_loader = get_mnist_dataloader(
            batch_size=eval_batch_size,
            split="eval_train",
            indices=train_indices,
            do_corrupt=do_corrupt,
        )
        valid_loader = get_mnist_dataloader(
            batch_size=eval_batch_size,
            split="valid",
            indices=valid_indices,
            do_corrupt=False,
        )
    elif data_name == "fmnist":
        train_loader = get_fmnist_dataloader(
            batch_size=train_batch_size,
            split="train",
            indices=train_indices,
            do_corrupt=do_corrupt,
        )
        eval_train_loader = get_fmnist_dataloader(
            batch_size=eval_batch_size,
            split="eval_train",
            indices=train_indices,
            do_corrupt=do_corrupt,
        )
        valid_loader = get_fmnist_dataloader(
            batch_size=eval_batch_size,
            split="valid",
            indices=valid_indices,
            do_corrupt=False,
        )
    else:
        raise ValueError(f"Unknown data_name: {data_name}.")
    return train_loader, eval_train_loader, valid_loader


def get_mnist_dataloader(
    batch_size: int,
    split: str = "train",
    indices: List[int] = None,
    do_corrupt: bool = False,
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

    if split in ["train", "eval_train"]:
        # Use approximately 10% of the training dataset.
        subset_size = 128 * 48
        dataset.data = dataset.data[:subset_size]
        dataset.targets = dataset.targets[:subset_size]
    else:
        # Use 1024 validation data points.
        subset_size = 1024
        dataset.data = dataset.data[:subset_size]
        dataset.targets = dataset.targets[:subset_size]

    if do_corrupt:
        if split == "valid":
            raise NotImplementedError(
                "Performing corruption on the validation dataset is not supported."
            )
        num_corrupt = math.ceil(len(dataset) * 0.1)
        original_targets = copy.deepcopy(dataset.targets[:num_corrupt])
        new_targets = torch.randint(
            0,
            10,
            size=original_targets[:num_corrupt].shape,
            generator=torch.Generator().manual_seed(0),
        )
        offsets = torch.randint(
            1,
            9,
            size=new_targets[new_targets == original_targets].shape,
            generator=torch.Generator().manual_seed(0),
        )
        new_targets[new_targets == original_targets] = (
            new_targets[new_targets == original_targets] + offsets
        ) % 10
        assert (new_targets == original_targets).sum() == 0
        dataset.targets[:num_corrupt] = new_targets

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=split == "train",
        batch_size=batch_size,
        num_workers=0,
        drop_last=split == "train",
    )


def get_fmnist_dataloader(
    batch_size: int,
    split: str = "train",
    indices: List[int] = None,
    do_corrupt: bool = False,
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

    if split in ["train", "eval_train"]:
        # Use approximately 10% of the training dataset.
        subset_size = 128 * 48
        dataset.data = dataset.data[:subset_size]
        dataset.targets = dataset.targets[:subset_size]
    else:
        # Use 1024 validation data points.
        subset_size = 1024
        dataset.data = dataset.data[:subset_size]
        dataset.targets = dataset.targets[:subset_size]

    if do_corrupt:
        if split == "valid":
            raise NotImplementedError(
                "Performing corruption on the validation dataset is not supported."
            )
        num_corrupt = math.ceil(len(dataset) * 0.1)
        original_targets = copy.deepcopy(dataset.targets[:num_corrupt])
        new_targets = torch.randint(
            0,
            10,
            size=original_targets[:num_corrupt].shape,
            generator=torch.Generator().manual_seed(0),
        )
        offsets = torch.randint(
            1,
            9,
            size=new_targets[new_targets == original_targets].shape,
            generator=torch.Generator().manual_seed(0),
        )
        new_targets[new_targets == original_targets] = (
            new_targets[new_targets == original_targets] + offsets
        ) % 10
        assert (new_targets == original_targets).sum() == 0
        dataset.targets[:num_corrupt] = new_targets

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=split == "train",
        batch_size=batch_size,
        num_workers=0,
        drop_last=split == "train",
    )
