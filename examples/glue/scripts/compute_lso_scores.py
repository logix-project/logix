import math
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.glue.pipeline import construct_model, get_hyperparameters, get_loaders
from examples.glue.scripts.train import train
from examples.utils import clear_gpu_cache, save_tensor

BASE_PATH = "../files"
BASE_REPEAT = 50
INDIV_REPEAT = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lso_evaluate(
    model: nn.Module, loader: torch.utils.data.DataLoader
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        dm_margins_lst = []
        trak_margins_lst = []
        losses_lst = []
        for batch in loader:
            outputs = model(
                batch["input_ids"].to(device=DEVICE),
                batch["token_type_ids"].to(device=DEVICE),
                batch["attention_mask"].to(device=DEVICE),
            )
            labels = batch["labels"].to(device=DEVICE)

            # Compute the loss.
            losses = F.cross_entropy(outputs, labels, reduction="none")
            losses_lst.append(losses.cpu())

            # Compute Datamodels margin.
            out_clone = outputs.clone()
            class_logits = out_clone[torch.arange(out_clone.shape[0]), labels].clone()
            out_clone[torch.arange(out_clone.shape[0]), labels] = -1000
            next_classes = out_clone.argmax(1)
            class_logits -= out_clone[torch.arange(out_clone.shape[0]), next_classes]
            dm_margins_lst.append(class_logits.cpu())

            # Compute TRAK margin.
            bindex = torch.arange(outputs.shape[0]).to(device="cpu", non_blocking=False)
            logits_correct = outputs[bindex, labels]
            out_clone = outputs.clone()
            out_clone[bindex, labels] = torch.tensor(
                -torch.inf, device="cpu", dtype=outputs.dtype
            )
            margins = logits_correct - out_clone.logsumexp(dim=-1)
            trak_margins_lst.append(margins.cpu())

        all_dm_margins = torch.cat(dm_margins_lst)
        all_trak_margins = torch.cat(trak_margins_lst)
        all_losses = torch.cat(losses_lst)
    return all_dm_margins, all_trak_margins, all_losses


def create_necessary_folders(data_name: str) -> None:
    os.makedirs("../files/", exist_ok=True)
    os.makedirs("../files/lso_scores/", exist_ok=True)
    os.makedirs(f"../files/lso_scores/data_{data_name}/", exist_ok=True)


def get_base_scores_path(data_name: str) -> str:
    return f"../files/lso_scores/data_{data_name}/base.pt"


def get_scores_path(data_name: str, alpha: float) -> str:
    return f"../files/lso_scores/data_{data_name}/{str(round(alpha, 3))}_{INDIV_REPEAT}r_{BASE_REPEAT}br.pt"


def lso_train(
    data_name: str,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    num_repeat: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hyper_dict = get_hyperparameters(data_name)
    lr = hyper_dict["lr"]
    wd = hyper_dict["wd"]

    dm_margins_lst = []
    trak_margins_lst = []
    losses_lst = []
    for _ in range(num_repeat):
        model = construct_model(data_name=data_name).to(DEVICE)
        model = train(
            model=model,
            loader=train_loader,
            lr=lr,
            weight_decay=wd,
        )
        dm_margins, trak_margins, losses = lso_evaluate(
            model,
            valid_loader,
        )
        dm_margins_lst.append(dm_margins)
        trak_margins_lst.append(trak_margins)
        losses_lst.append(losses)
        del model, dm_margins, trak_margins, losses
        clear_gpu_cache()

    avg_dm_margin = np.mean(np.array(dm_margins_lst), 0)
    avg_trak_margin = np.mean(np.array(trak_margins_lst), 0)
    avg_loss = np.mean(np.array(losses_lst), 0)
    return avg_dm_margin, avg_trak_margin, avg_loss


def compute_base_score(
    data_name: str,
) -> None:
    clear_gpu_cache()
    create_necessary_folders(data_name=data_name)
    file_name = get_base_scores_path(data_name=data_name)
    if os.path.exists(file_name):
        print("Base scores already computed!")
        return

    train_loader, _, valid_loader = get_loaders(
        data_name=data_name,
        train_indices=None,
        do_corrupt=False,
    )
    avg_dm_margins, avg_trak_margins, avg_losses = lso_train(
        data_name=data_name,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_repeat=BASE_REPEAT,
    )
    results = {
        "base_dm_margins": avg_dm_margins,
        "base_trak_margins": avg_trak_margins,
        "base_losses": avg_losses,
    }
    save_tensor(
        results,
        file_name=file_name,
        overwrite=False,
    )


def compute_score(
    data_name: str,
    alpha: float,
    num_masks: int = 120,
) -> None:
    clear_gpu_cache()
    create_necessary_folders(data_name=data_name)
    file_name = get_scores_path(data_name=data_name, alpha=alpha)
    if os.path.exists(file_name):
        print(f"LSO score for {file_name} already exists!")
        return

    _, eval_train_loader, valid_loader = get_loaders(
        data_name=data_name,
        train_indices=None,
        do_corrupt=False,
    )
    num_train = len(eval_train_loader.dataset)
    del eval_train_loader

    base_file_name = get_base_scores_path(data_name=data_name)
    if not os.path.exists(base_file_name):
        raise RuntimeError(f"Cannot find base scores at {base_file_name}.")
    base_results = torch.load(base_file_name)
    base_dm_margins = base_results["base_dm_margins"]
    base_trak_margins = base_results["base_trak_margins"]
    base_losses = base_results["base_losses"]
    del base_results, base_file_name
    print("Loaded base scores.")

    num_data_to_remove = math.ceil(num_train * alpha)
    print(f"Removing {num_data_to_remove} points (out of {num_train} points).")
    if num_data_to_remove == 1:
        print("Removing a single data point!")
    masks_lst = []
    binary_masks_lst = []
    for _ in range(num_masks):
        mask = np.random.choice(
            np.arange(num_train), replace=False, size=num_train - num_data_to_remove
        )
        assert len(mask) == num_train - num_data_to_remove
        masks_lst.append(mask)
        binary_mask = torch.zeros(
            num_train, device="cpu", dtype=torch.int8, requires_grad=False
        )
        binary_mask[mask] = 1
        assert binary_mask.sum() == num_train - num_data_to_remove
        binary_masks_lst.append(binary_mask)

    dm_margin_diff_lst = []
    trak_margin_diff_lst = []
    loss_diff_lst = []
    for i in range(num_masks):
        print(f"Training {i}th model.")
        start_time = time.time()

        masked_train_loader, _, _ = get_loaders(
            data_name=data_name,
            train_indices=masks_lst[i],
            do_corrupt=False,
        )
        assert len(masked_train_loader.dataset) == num_train - num_data_to_remove

        avg_dm_margins, avg_trak_margins, avg_losses = lso_train(
            data_name=data_name,
            train_loader=masked_train_loader,
            valid_loader=valid_loader,
            num_repeat=INDIV_REPEAT,
        )

        dm_margin_diff_lst.append(avg_dm_margins - base_dm_margins)
        trak_margin_diff_lst.append(avg_trak_margins - base_trak_margins)
        loss_diff_lst.append(avg_losses - base_losses)
        del masked_train_loader, avg_dm_margins, avg_trak_margins, avg_losses

        print(f"Took {time.time() - start_time} seconds.")

    results = {
        "base_dm_margins": base_dm_margins,
        "base_trak_margins": base_trak_margins,
        "base_losses": base_losses,
        "dm_margin_diffs": np.array(dm_margin_diff_lst),
        "trak_margin_diffs": np.array(trak_margin_diff_lst),
        "loss_diffs": np.array(loss_diff_lst),
        "masks": np.array(masks_lst),
        "binary_masks": np.array(binary_masks_lst),
    }
    save_tensor(results, file_name=file_name, overwrite=False)


if __name__ == "__main__":
    import sys

    alpha_lst1 = [0.7, 0.5, 0.3]
    alpha_lst2 = [0.1, (1 / 100_000)]
    try:
        option = int(sys.argv[1])
        if option == 1:
            compute_base_score(data_name="qnli")
            for alpha in alpha_lst1:
                compute_score(data_name="qnli", alpha=alpha)
        elif option == 2:
            compute_base_score(data_name="sst2")
            for alpha in alpha_lst1:
                compute_score(data_name="sst2", alpha=alpha)
        elif option == 3:
            compute_base_score(data_name="qnli")
            for alpha in alpha_lst2:
                compute_score(data_name="qnli", alpha=alpha)
        elif option == 4:
            compute_base_score(data_name="sst2")
            for alpha in alpha_lst2:
                compute_score(data_name="sst2", alpha=alpha)
        elif option == 5:
            compute_base_score(data_name="rte")
            for alpha in alpha_lst1:
                compute_score(data_name="rte", alpha=alpha)
        elif option == 6:
            compute_base_score(data_name="rte")
            for alpha in alpha_lst2:
                compute_score(data_name="rte", alpha=alpha)
        else:
            raise NotImplementedError(f"Not an available option {option}.")

    except IndexError:
        compute_base_score(data_name="qnli")
        compute_score(
            data_name="qnli",
            alpha=0.5,
            num_masks=2,
        )
