import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from pipeline import construct_mlp, get_hyperparameters, get_loaders
from utils import clear_gpu_cache, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    model_id: int = 0,
    lr: float = 0.1,
    weight_decay=1e-4,
    save_name: Optional[str] = None,
) -> nn.Module:
    save = save_name is not None
    if save:
        os.makedirs(f"files/checkpoints/{model_id}", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"files/checkpoints/{model_id}/{save_name}_epoch_0.pt",
        )
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss()
    epochs = 20

    model.train()
    for epoch in range(1, epochs + 1):
        for images, labels in loader:
            images, labels = images.to(device=DEVICE), labels.to(device=DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        if save:
            torch.save(
                model.state_dict(),
                f"files/checkpoints/{model_id}/{save_name}_epoch_{epoch}.pt",
            )
    return model


def evaluate(
    model: nn.Module, loader: torch.utils.data.DataLoader
) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        total_loss, total_correct, total_num = 0.0, 0.0, 0.0
        for images, labels in loader:
            images = images.to(device=DEVICE)
            labels = labels.to(device=DEVICE)
            outputs = model(images)
            total_loss += F.cross_entropy(outputs, labels, reduction="sum").cpu().item()
            total_correct += outputs.argmax(1).eq(labels).sum().cpu().item()
            total_num += images.shape[0]
    return total_loss / total_num, total_correct / total_num


def main(
    data_name: str = "mnist",
    num_train: int = 5,
) -> None:
    os.makedirs("files", exist_ok=True)
    os.makedirs("files/checkpoints", exist_ok=True)

    train_loader, _, valid_loader = get_loaders(
        data_name=data_name,
    )
    hyper_dict = get_hyperparameters(data_name)
    lr = hyper_dict["lr"]
    wd = hyper_dict["wd"]

    save_name = data_name
    for i in range(num_train):
        print(f"Training {i}th model ...")
        start_time = time.time()

        set_seed(i)
        model = construct_mlp().to(DEVICE)
        model = train(
            model=model,
            loader=train_loader,
            lr=lr,
            weight_decay=wd,
            model_id=i,
            save_name=save_name,
        )

        loss, acc = evaluate(model, valid_loader)
        print(f"Loss: {loss}")
        print(f"Accuracy: {acc}")
        del model
        clear_gpu_cache()
        print(f"Took {time.time() - start_time} seconds.")


if __name__ == "__main__":
    main(data_name="mnist", num_train=1)
    main(data_name="fmnist", num_train=1)
