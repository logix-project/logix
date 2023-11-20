"""Trains the network using the similar configurations from the following papers (other papers might have
used this configuration as well):
- https://arxiv.org/pdf/1703.04730.pdf
- https://arxiv.org/pdf/2209.05364.pdf
- https://arxiv.org/pdf/2008.03703.pdf

Note that:
- We use 10% of the MNIST & FMNIST datasets.
- Grid searches are performed on LR and WD.
    - MNIST test accuracy: ~95.5%
    - FMNIST test accuracy: ~83.9%
"""

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import os
import torch

from utils import (
    get_mnist_dataloader,
    construct_mlp,
    get_fmnist_dataloader,
    set_seed,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    loader,
    lr=0.1,
    epochs=10,
    momentum=0.9,
    weight_decay=1e-4,
    label_smoothing=0.0,
    save_name="default",
    model_id=0,
    save=True,
):
    if save:
        torch.save(model.state_dict(), f"checkpoints/{save_name}_{model_id}_epoch_0.pt")
    optimizer = SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for epoch in range(epochs):
        # We use consistent data ordering when training.
        set_seed(model_id * 10_061 + epoch + 1)
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
        if save:
            torch.save(
                model.state_dict(),
                f"checkpoints/{save_name}_{model_id}_epoch_{epoch}.pt",
            )
    return model


def main(dataset="mnist"):
    os.makedirs("checkpoints", exist_ok=True)

    if dataset == "mnist":
        train_loader = get_mnist_dataloader(
            batch_size=128, split="train", shuffle=True, subsample=True
        )
        valid_loader = get_mnist_dataloader(
            batch_size=512, split="valid", shuffle=False
        )
    else:
        train_loader = get_fmnist_dataloader(
            batch_size=128, split="train", shuffle=True, subsample=True
        )
        valid_loader = get_fmnist_dataloader(
            batch_size=512, split="valid", shuffle=False
        )

    for i in range(10):
        print(f"Model {i}")
        model = construct_mlp(seed=i).to(DEVICE)

        # I performed a grid search with the following search space:
        # LR: [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
        # WD: [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 0]
        if dataset == "mnist":
            model = train(
                model,
                train_loader,
                lr=0.1,
                weight_decay=0.001,
                save_name=dataset,
                model_id=i,
            )
        else:
            model = train(
                model,
                train_loader,
                lr=0.1,
                weight_decay=0.0001,
                save_name=dataset,
                model_id=i,
            )

        model.eval()
        with torch.no_grad():
            total_correct, total_num = 0.0, 0.0
            for ims, labs in valid_loader:
                ims = ims.to(DEVICE)
                ims = ims.reshape(ims.shape[0], -1)
                labs = labs.to(DEVICE)
                out = model(ims)
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]
            print(f"Accuracy: {total_correct / total_num * 100:.1f}%")


if __name__ == "__main__":
    main(dataset="mnist")
    main(dataset="fmnist")
