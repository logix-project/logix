import random

import numpy as np
import torch
import torch.nn as nn
import torchvision


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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
