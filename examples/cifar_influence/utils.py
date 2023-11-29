# We use an example from the TRAK repository:
# https://github.com/MadryLab/trak/blob/main/examples/cifar_quickstart.ipynb.


import torch
import torchvision
import numpy as np
import random

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def construct_rn9(num_classes=10, seed=0):
    set_seed(seed)
    def conv_bn(
        channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1
    ):
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
            torch.nn.ReLU(inplace=True),
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


def get_cifar10_dataloader(
    batch_size=256,
    num_workers=8,
    split="train",
    shuffle=False,
    augment=True,
    drop_last=False,
    subsample = False,
    indices=None,
):
    if augment:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomAffine(0),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)
                ),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)
                ),
            ]
        )

    is_train = split == "train"
    dataset = torchvision.datasets.CIFAR10(
        root="/tmp/cifar/", download=True, train=is_train, transform=transforms
    )

    if subsample and split == "train" and indices is None:
        dataset = torch.utils.data.Subset(dataset, np.arange(6_000))

    if indices is not None:
        if subsample and split == "train":
            print("Overriding `subsample` argument as `indices` was provided.")
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last
    )

    return loader
