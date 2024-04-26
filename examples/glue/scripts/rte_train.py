import os
import time
from typing import Optional, Tuple

import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from examples.glue.pipeline import construct_model, get_hyperparameters, get_loaders
from examples.utils import clear_gpu_cache, save_tensor, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


def train(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    lr: float,
    weight_decay: float,
    model_id: int = 0,
    save_name: Optional[str] = None,
    epochs: int = 3,
) -> nn.Module:
    save = save_name is not None
    if save:
        os.makedirs(f"../files/checkpoints/{save_name}/", exist_ok=True)
        os.makedirs(
            f"../files/checkpoints/{save_name}/model_{model_id}/", exist_ok=True
        )
        save_tensor(
            model.state_dict(),
            f"../files/checkpoints/{save_name}/model_{model_id}/iter_0.pt",
            overwrite=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss(reduction="mean")

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
                    f"../files/checkpoints/{save_name}/model_{model_id}/iter_{num_iter}.pt",
                    overwrite=True,
                )
                save_tensor(
                    optimizer.state_dict(),
                    f"../files/checkpoints/{save_name}/model_{model_id}/iter_{num_iter}_optimizer.pt",
                    overwrite=True,
                )
    return model


def model_evaluate(
    model: nn.Module, loader: torch.utils.data.DataLoader
) -> Tuple[float, float]:
    model.eval()
    # Task name does not really matter here.
    metric = evaluate.load("glue", "qnli")
    total_loss, total_num = 0.0, 0.0
    for step, batch in enumerate(loader):
        with torch.no_grad():
            outputs = model(
                batch["input_ids"].to(device=DEVICE),
                batch["token_type_ids"].to(device=DEVICE),
                batch["attention_mask"].to(device=DEVICE),
            )
            labels = batch["labels"].to(device=DEVICE)
        total_loss += F.cross_entropy(outputs, labels, reduction="sum").cpu().item()
        total_num += batch["input_ids"].shape[0]
        predictions = outputs.argmax(dim=-1)
        metric.add_batch(
            predictions=predictions,
            references=labels,
        )
    eval_metric = metric.compute()
    return total_loss / total_num, eval_metric["accuracy"]


def main(data_name: str = "rte", num_train: int = 50, do_corrupt: bool = False) -> None:
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

        last_epoch_name = f"../files/checkpoints/{save_name}/model_{i}/iter_228.pt"
        if os.path.exists(last_epoch_name):
            print("Already exists!")
        else:
            set_seed(i)
            model = construct_model(data_name=data_name).to(device=DEVICE)
            model = train(
                model=model,
                loader=train_loader,
                lr=lr,
                weight_decay=wd,
                model_id=i,
                save_name=save_name,
            )
            _, valid_acc = model_evaluate(model=model, loader=valid_loader)
            print(f"Validation Accuracy: {valid_acc}")
            del model
            clear_gpu_cache()

        print(f"Took {time.time() - start_time} seconds.")


if __name__ == "__main__":
    main(data_name="rte", do_corrupt=False, num_train=1)
