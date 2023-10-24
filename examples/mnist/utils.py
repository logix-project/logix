import random
import numpy as np
import torch
import torchvision
import torch.nn as nn


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def construct_mlp(num_inputs=784, num_classes=10, seed=0):
    # Configurations used in the "influence memorization" paper:
    # https://github.com/google-research/heldout-influence-estimation/blob/master/mnist-example/mnist_infl_mem.py.
    set_seed(seed)
    model = torch.nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, 512, bias=True),
        nn.ReLU(),
        nn.Linear(512, 256, bias=True),
        nn.ReLU(),
        nn.Linear(256, num_classes, bias=True),
    )
    return model


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


def get_fmnist_dataloader(
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
            torchvision.transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    is_train = split == "train"
    dataset = torchvision.datasets.FashionMNIST(
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


if __name__ == "__main__":
    # Verifying if datasets look reasonable.
    import matplotlib.pyplot as plt

    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    loader = get_mnist_dataloader(batch_size=16, shuffle=True, subsample=True)
    data_iter = iter(loader)
    data = next(data_iter)
    imshow(torchvision.utils.make_grid(data[0], padding=2))

    loader = get_fmnist_dataloader(batch_size=16, shuffle=True, subsample=True)
    data_iter = iter(loader)
    data = next(data_iter)
    imshow(torchvision.utils.make_grid(data[0], padding=2))
