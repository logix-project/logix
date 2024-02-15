import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from examples.pipeline import construct_model, get_hyperparameters, get_loaders
from examples.utils import clear_gpu_cache, save_tensor, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    lr: float,
    weight_decay: float,
    epochs: int = 20,
    model_id: int = 0,
    save_name: Optional[str] = None,
) -> nn.Module:
    save = save_name is not None
    if save:
        os.makedirs(f"../files/checkpoints/{save_name}/", exist_ok=True)
        os.makedirs(
            f"../files/checkpoints/{save_name}/model_{model_id}/", exist_ok=True
        )
        save_tensor(
            model.state_dict(),
            f"../files/checkpoints/{save_name}/model_{model_id}/epoch_0.pt",
            overwrite=True,
        )
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss(reduction="mean")

    model.train()
    for epoch in range(1, epochs + 1):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        if save:
            save_tensor(
                model.state_dict(),
                f"../files/checkpoints/{save_name}/model_{model_id}/epoch_{epoch}.pt",
                overwrite=True,
            )
    return model


def evaluate(
    model: nn.Module, loader: torch.utils.data.DataLoader
) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        total_loss, total_correct, total_num = 0.0, 0.0, 0.0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            total_loss += F.cross_entropy(outputs, labels, reduction="sum").cpu().item()
            total_correct += outputs.argmax(1).eq(labels).sum().cpu().item()
            total_num += images.shape[0]
    return total_loss / total_num, total_correct / total_num


def main(
    data_name: str = "mnist", num_train: int = 50, do_corrupt: bool = False
) -> None:
    os.makedirs("../files", exist_ok=True)
    os.makedirs("../files/checkpoints", exist_ok=True)

    train_loader, _, valid_loader = get_loaders(
        data_name=data_name,
        do_corrupt=do_corrupt,
    )
    hyper_dict = get_hyperparameters(data_name)
    lr = hyper_dict["lr"]
    wd = hyper_dict["wd"]

    save_name = f"data_{data_name}"
    if do_corrupt:
        save_name += "_corrupted"

    for i in range(num_train):
        print(f"Training {i}th model.")
        start_time = time.time()

        last_epoch_name = f"../files/checkpoints/data_{data_name}/model_{i}/epoch_20.pt"
        if os.path.exists(last_epoch_name):
            print("Already exists!")
        else:
            set_seed(i)
            model = construct_model(data_name).to(DEVICE)
            train(
                model=model,
                loader=train_loader,
                lr=lr,
                weight_decay=wd,
                model_id=i,
                save_name=save_name,
            )
            valid_loss, valid_acc = evaluate(model, valid_loader)
            print(f"Validation Loss: {valid_loss}.")
            print(f"Validation Accuracy: {valid_acc}.")
            del model
            clear_gpu_cache()

        print(f"Took {time.time() - start_time} seconds.")


if __name__ == "__main__":
    data_names = ["mnist", "fmnist"]
    for dn in data_names:
        main(data_name=dn, do_corrupt=False, num_train=10)
        main(data_name=dn, do_corrupt=False, num_train=10)
