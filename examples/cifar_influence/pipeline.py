import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision


class Mul(torch.nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


def construct_model(data_name: str = "cifar2") -> nn.Module:
    assert data_name in ["cifar2", "cifar10"]
    num_classes = 2 if data_name == "cifar2" else 10
    del data_name

    def conv_bn(
        channels_in: int,
        channels_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups=1,
    ) -> nn.Module:
        assert groups == 1
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(),
        )

    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2),
    )
    return model


def get_hyperparameters(data_name: str) -> Dict[str, float]:
    wd = 0.001
    if data_name == "cifar2":
        lr = 0.5
        epochs = 100
    elif data_name == "cifar10":
        lr = 0.4
        epochs = 25
    else:
        raise NotImplementedError()
    return {"lr": lr, "wd": wd, "epochs": epochs}


# Configurations from:
# https://github.com/mosaicml/composer/blob/d952e1da11256c430a8291cd39d57783d414b391/composer/datasets/cifar.py.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.247, 0.243, 0.261)


def get_loaders(
    data_name: str,
    eval_batch_size: int = 2048,
    train_indices: Optional[List[int]] = None,
    valid_indices: Optional[List[int]] = None,
    do_corrupt: bool = False,
    num_workers: int = 4,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    assert data_name in ["cifar2", "cifar10"]
    train_batch_size = 512

    if data_name == "cifar2":
        train_loader = get_cifar2_dataloader(
            batch_size=train_batch_size,
            split="train",
            indices=train_indices,
            do_corrupt=do_corrupt,
            num_workers=num_workers,
        )
        eval_train_loader = get_cifar2_dataloader(
            batch_size=eval_batch_size,
            split="eval_train",
            indices=train_indices,
            do_corrupt=do_corrupt,
            num_workers=0,
        )
        valid_loader = get_cifar2_dataloader(
            batch_size=eval_batch_size,
            split="valid",
            indices=valid_indices,
            do_corrupt=False,
            num_workers=0,
        )
    else:
        train_loader = get_cifar10_dataloader(
            batch_size=train_batch_size,
            split="train",
            indices=train_indices,
            do_corrupt=do_corrupt,
            num_workers=num_workers,
        )
        eval_train_loader = get_cifar10_dataloader(
            batch_size=eval_batch_size,
            split="eval_train",
            indices=train_indices,
            do_corrupt=do_corrupt,
            num_workers=0,
        )
        valid_loader = get_cifar10_dataloader(
            batch_size=eval_batch_size,
            split="valid",
            indices=valid_indices,
            do_corrupt=False,
            num_workers=0,
        )
    return train_loader, eval_train_loader, valid_loader


def get_eval_train_loader_with_aug(
    data_name: str,
    eval_batch_size: int = 2048,
    train_indices: Optional[List[int]] = None,
    do_corrupt: bool = False,
) -> torch.utils.data.DataLoader:
    assert data_name in ["cifar2", "cifar10"]

    if data_name == "cifar2":
        eval_train_loader = get_cifar2_dataloader(
            batch_size=eval_batch_size,
            split="eval_train_with_aug",
            indices=train_indices,
            do_corrupt=do_corrupt,
            num_workers=0,
        )
    else:
        eval_train_loader = get_cifar10_dataloader(
            batch_size=eval_batch_size,
            split="eval_train_with_aug",
            indices=train_indices,
            do_corrupt=do_corrupt,
            num_workers=0,
        )
    return eval_train_loader


def get_cifar2_dataloader(
    batch_size: int,
    split: str,
    indices: List[int] = None,
    do_corrupt: bool = False,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    assert split in ["train", "eval_train", "eval_train_with_aug", "valid"]

    if split in ["train", "eval_train_with_aug"]:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    dataset = torchvision.datasets.CIFAR10(
        root="/tmp/cifar/",
        download=True,
        train=split in ["train", "eval_train", "eval_train_with_aug"],
        transform=transforms,
    )
    # We only select "cats" or "dogs" classes.
    valid_idxs = np.where(
        (np.array(dataset.targets) == 3) | (np.array(dataset.targets) == 5)
    )[0]

    # Binarize the targets.
    new_targets = []
    for target in dataset.targets:
        if target == 3:
            new_targets.append(0)
        elif target == 5:
            new_targets.append(1)
        else:
            new_targets.append(target)
    dataset.targets = new_targets

    dataset.data = dataset.data[valid_idxs]
    dataset.targets = list(np.array(dataset.targets)[valid_idxs])
    dataset.classes = ["cat", "dog"]

    if do_corrupt:
        if split == "valid":
            raise NotImplementedError(
                "Performing corruption on the validation dataset is not supported."
            )
        num_corrupt = math.ceil(len(dataset) * 0.1)
        original_targets = copy.deepcopy(dataset.targets)
        corrupted_targets = list((np.array(original_targets[:num_corrupt]) + 1) % 2)
        dataset.targets[:num_corrupt] = corrupted_targets

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=split == "train",
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=split == "train",
        pin_memory=True,
    )


def get_cifar10_dataloader(
    batch_size: int,
    split: str = "train",
    indices: List[int] = None,
    do_corrupt: bool = False,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    assert split in ["train", "eval_train", "eval_train_with_aug", "valid"]

    if split in ["train", "eval_train_with_aug"]:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    dataset = torchvision.datasets.CIFAR10(
        root="/tmp/cifar/",
        download=True,
        train=split in ["train", "eval_train", "eval_train_with_aug"],
        transform=transforms,
    )

    if do_corrupt:
        if split == "valid":
            raise NotImplementedError(
                "Performing corruption on the validation dataset is not supported."
            )
        num_corrupt = math.ceil(len(dataset) * 0.1)
        original_targets = np.array(copy.deepcopy(dataset.targets[:num_corrupt]))
        new_targets = torch.randint(
            0,
            10,
            size=original_targets[:num_corrupt].shape,
            generator=torch.Generator().manual_seed(0),
        ).numpy()
        offsets = torch.randint(
            1,
            9,
            size=new_targets[new_targets == original_targets].shape,
            generator=torch.Generator().manual_seed(0),
        ).numpy()
        new_targets[new_targets == original_targets] = (
            new_targets[new_targets == original_targets] + offsets
        ) % 10
        assert (new_targets == original_targets).sum() == 0
        dataset.targets[:num_corrupt] = list(new_targets)

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=split == "train",
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=split == "train",
        pin_memory=True,
    )
