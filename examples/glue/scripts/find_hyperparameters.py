import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from examples.glue.pipeline import construct_model, get_loaders
from examples.glue.scripts.train import model_evaluate
from examples.utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


def train_with_evaluation(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    eval_train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    lr: float,
    epochs: int,
    weight_decay: float,
    save_intermediate: bool = True,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss(reduction="mean")

    train_loss_lst, train_acc_lst = [], []
    valid_loss_lst, valid_acc_lst = [], []
    if save_intermediate:
        train_loss, train_acc = model_evaluate(model, eval_train_loader)
        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)

        valid_loss, valid_acc = model_evaluate(model, valid_loader)
        valid_loss_lst.append(valid_loss)
        valid_acc_lst.append(valid_acc)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                batch["input_ids"].to(device=DEVICE),
                batch["token_type_ids"].to(device=DEVICE),
                batch["attention_mask"].to(device=DEVICE),
            )
            loss = loss_fn(outputs, batch["labels"].to(device=DEVICE))
            loss.backward()
            optimizer.step()

        if save_intermediate or epoch == epochs - 1:
            train_loss, train_acc = model_evaluate(model, eval_train_loader)
            train_loss_lst.append(train_loss)
            train_acc_lst.append(train_acc)

            valid_loss, valid_acc = model_evaluate(model, valid_loader)
            valid_loss_lst.append(valid_loss)
            valid_acc_lst.append(valid_acc)

    return train_loss_lst, train_acc_lst, valid_loss_lst, valid_acc_lst


def main(data_name: str, target_epochs: int = 3) -> None:
    lr_lst = [1e-4, 2e-5, 3e-5, 1e-5]  # Reduced search space.
    wd_lst = [1e-2, 1e-3, 1e-4, 1e-5, 0.0]  # Reduced search space.

    train_loader, eval_train_loader, valid_loader = get_loaders(data_name=data_name)

    best_acc = None
    best_lr = None
    best_wd = None
    for lr in lr_lst:
        for wd in wd_lst:
            start_time = time.time()

            total_accs = []
            for seed in range(5):
                set_seed(seed)
                model = construct_model(data_name).to(DEVICE)
                _, _, _, valid_acc = train_with_evaluation(
                    model=model,
                    train_loader=train_loader,
                    eval_train_loader=eval_train_loader,
                    valid_loader=valid_loader,
                    lr=lr,
                    weight_decay=wd,
                    epochs=target_epochs,
                    save_intermediate=False,
                )
                assert len(valid_acc) == 1
                total_accs.append(valid_acc[-1])

            final_acc = np.mean(np.array(total_accs))
            if np.isnan(final_acc):
                final_acc = -np.Inf

            if best_acc is None or final_acc > best_acc:
                print("Found best results!")
                print(f"Accuracy: {final_acc}")
                best_acc = final_acc
                best_lr = lr
                best_wd = wd

            print(f"Took {time.time() - start_time} seconds.")

    print(f"Dataset: {data_name}")
    print(f"Best LR, WD: {best_lr}, {best_wd}")
    print(f"Best Acc: {best_acc}")
    total_epochs = 5
    set_seed(0)
    model = construct_model(data_name=data_name).to(DEVICE)
    train_losses, train_accs, valid_losses, valid_accs = train_with_evaluation(
        model=model,
        train_loader=train_loader,
        eval_train_loader=eval_train_loader,
        valid_loader=valid_loader,
        lr=best_lr,
        weight_decay=best_wd,
        epochs=total_epochs,
        save_intermediate=True,
    )

    plt.plot(list(range(total_epochs + 1)), train_losses, label="Train")
    plt.plot(list(range(total_epochs + 1)), valid_losses, label="Valid")
    plt.title(f"{data_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.clf()

    plt.plot(list(range(total_epochs + 1)), train_accs, label="Train")
    plt.plot(list(range(total_epochs + 1)), valid_accs, label="Valid")
    plt.title(f"{data_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import sys

    try:
        option = int(sys.argv[1])
        if option == 1:
            main(data_name="sst2", target_epochs=3)
        elif option == 2:
            main(data_name="qnli", target_epochs=3)
        elif option == 3:
            main(data_name="rte", target_epochs=3)
        else:
            raise NotImplementedError(f"Not an available option {option}.")

    except IndexError:
        main(data_name="sst2", target_epochs=3)
        main(data_name="qnli", target_epochs=3)
        main(data_name="rte", target_epochs=3)
