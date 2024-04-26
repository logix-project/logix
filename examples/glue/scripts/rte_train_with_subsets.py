import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from examples.glue.pipeline import construct_model, get_hyperparameters, get_loaders
from examples.utils import clear_gpu_cache, save_tensor, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_with_subsets(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    lr: float,
    weight_decay: float,
    model_id: int = 0,
    save_name: Optional[str] = None,
) -> nn.Module:
    save = save_name is not None
    if save:
        os.makedirs(f"../files/subset_checkpoints/{save_name}/", exist_ok=True)
        os.makedirs(
            f"../files/subset_checkpoints/{save_name}/model_{model_id}/", exist_ok=True
        )
        save_tensor(
            model.state_dict(),
            f"../files/subset_checkpoints/{save_name}/model_{model_id}/iter_0.pt",
            overwrite=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss(reduction="mean")
    epochs = 3

    model.train()
    num_iter = 0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(
                batch["input_ids"].to(device=DEVICE),
                batch["token_type_ids"].to(device=DEVICE),
                batch["attention_mask"].to(device=DEVICE),
            )
            loss = loss_fn(outputs, batch["labels"].to(device=DEVICE))
            loss.backward()
            optimizer.step()
            num_iter += 1

            if save and num_iter % 38 == 0:
                save_tensor(
                    model.state_dict(),
                    f"../files/subset_checkpoints/{save_name}/model_{model_id}/iter_{num_iter}.pt",
                    overwrite=True,
                )
                save_tensor(
                    optimizer.state_dict(),
                    f"../files/subset_checkpoints/{save_name}/model_{model_id}/iter_{num_iter}_optimizer.pt",
                    overwrite=True,
                )
    return model


def main(
    data_name: str = "qnli",
    num_train: int = 50,
) -> None:
    os.makedirs("../files", exist_ok=True)
    os.makedirs("../files/subset_checkpoints", exist_ok=True)
    os.makedirs(f"../files/subset_checkpoints/data_{data_name}", exist_ok=True)

    train_loader, eval_train_loader, valid_loader = get_loaders(
        data_name=data_name,
        do_corrupt=False,
    )
    train_size = len(eval_train_loader.dataset)

    hyper_dict = get_hyperparameters(data_name)
    lr = hyper_dict["lr"]
    wd = hyper_dict["wd"]

    alpha_list = [0.5]
    for alpha in alpha_list:
        alpha_str = str(round(alpha, 3))
        os.makedirs(
            f"../files/subset_checkpoints/data_{data_name}/alpha_{alpha_str}/",
            exist_ok=True,
        )

        set_seed(0)
        masks_lst = []
        num_data_to_remove = math.ceil(train_size * alpha)
        for _ in range(num_train):
            mask = np.random.choice(
                np.arange(train_size),
                replace=False,
                size=train_size - num_data_to_remove,
            )
            assert len(mask) == train_size - num_data_to_remove
            masks_lst.append(mask)

        for i in range(num_train):
            last_epoch_name = f"../files/subset_checkpoints/data_{data_name}/alpha_{alpha_str}/model_{i}/iter_228.pt"
            if os.path.exists(last_epoch_name):
                print("Already exists!")
            else:
                masked_train_loader, _, _ = get_loaders(
                    data_name=data_name,
                    train_indices=masks_lst[i],
                    do_corrupt=False,
                )

                set_seed(i)
                model = construct_model(data_name=data_name).to(device=DEVICE)
                train_with_subsets(
                    model=model,
                    loader=train_loader,
                    lr=lr,
                    weight_decay=wd,
                    model_id=i,
                    save_name=f"data_{data_name}/alpha_{alpha_str}",
                )
                del model
                clear_gpu_cache()


if __name__ == "__main__":
    main(data_name="rte", num_train=100)
