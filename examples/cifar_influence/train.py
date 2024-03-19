import argparse
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from examples.cifar_influence.pipeline import construct_model, get_hyperparameters, get_loaders, get_eval_train_loader_with_aug
from examples.utils import clear_gpu_cache, save_tensor, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


def train(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    lr: float,
    epochs: int,
    weight_decay: float,
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
    lr_peak_epoch = epochs // 5
    iters_per_epoch = len(loader)
    lr_schedule = np.interp(
        np.arange((epochs + 1) * iters_per_epoch),
        [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
        [0, 1, 0],
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    model.train()
    for epoch in range(1, epochs + 1):
        for images, labels in loader:
            images, labels = images.to(device=DEVICE), labels.to(device=DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        if save and epoch == epochs:
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
            images, labels = images.to(device=DEVICE), labels.to(device=DEVICE)
            outputs = model(images)
            total_loss += F.cross_entropy(outputs, labels, reduction="sum").cpu().item()
            total_correct += outputs.argmax(1).eq(labels).sum().cpu().item()
            total_num += images.shape[0]
    return total_loss / total_num, total_correct / total_num


def main(
    data_name: str = "cifar2", num_train: int = 50, do_corrupt: bool = False, model_id = 0
) -> None:
    os.makedirs("../files", exist_ok=True)
    os.makedirs("../files/checkpoints", exist_ok=True)

    train_loader, _, valid_loader = get_loaders(
        data_name=data_name,
        do_corrupt=do_corrupt,
        num_workers=4,
    )
    hyper_dict = get_hyperparameters(data_name=data_name)
    lr = hyper_dict["lr"]
    wd = hyper_dict["wd"]
    epochs = int(hyper_dict["epochs"])

    save_name = f"data_{data_name}"
    if do_corrupt:
        save_name += "_corrupted"

    print(f"Training model_id={model_id} model.")
    start_time = time.time()

    last_epoch_name = (
        f"../files/checkpoints/{save_name}/model_{model_id}/epoch_{epochs}.pt"
    )
    if os.path.exists(last_epoch_name):
        print("Already exists!")
    else:
        set_seed(model_id)
        model = construct_model(data_name=data_name).to(DEVICE)
        model = train(
            model=model,
            loader=train_loader,
            lr=lr,
            epochs=epochs,
            weight_decay=wd,
            model_id=model_id,
            save_name=save_name,
        )
        valid_loss, valid_acc = evaluate(model, valid_loader)
        print(f"Validation Loss: {valid_loss}")
        print(f"Validation Accuracy: {valid_acc}")
        del model
        clear_gpu_cache()

        print(f"Took {time.time() - start_time} seconds.")


if __name__ == "__main__":
    # create arg parse for model_id
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int, default=0)
    args = parser.parse_args()
    # main(data_name="cifar2", do_corrupt=False, num_train=5)
    main(data_name="cifar10", do_corrupt=False, num_train=10, model_id=args.model_id)
    # main(data_name="cifar2", do_corrupt=True, num_train=5)
    # main(data_name="cifar10", do_corrupt=True, num_train=5)
