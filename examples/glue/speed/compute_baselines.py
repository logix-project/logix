import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from experiments.compute_utils import (
    compute_gradient_similarity,
    compute_influence_function,
    compute_representation_similarity,
    compute_tracin,
    compute_trak,
    get_default_file_name,
)
from experiments.glue.pipeline import construct_model, get_hyperparameters, get_loaders
from experiments.glue.task import GlueExperimentTask, GlueWithLayerNormExperimentTask
from experiments.utils import save_tensor
from src.abstract_task import AbstractTask

BASE_PATH = "../files/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


def prepare_everything(
    data_name: str,
    model_id: int = 0,
) -> Tuple[
    nn.Module, torch.utils.data.DataLoader, torch.utils.data.DataLoader, AbstractTask
]:
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(f"{BASE_PATH}/data_{data_name}/", exist_ok=True)
    os.makedirs(f"{BASE_PATH}/data_{data_name}/model_{model_id}/", exist_ok=True)

    if data_name == "rte":
        _, eval_train_loader, _ = get_loaders(
            data_name=data_name,
            train_indices=None,
            valid_indices=None,
            eval_batch_size=32,
        )
        _, _, valid_loader = get_loaders(
            data_name=data_name,
            train_indices=None,
            valid_indices=None,
            eval_batch_size=16,
        )
    else:
        _, eval_train_loader, _ = get_loaders(
            data_name=data_name,
            train_indices=None,
            valid_indices=list(range(512)),
            eval_batch_size=32,
        )
        _, _, valid_loader = get_loaders(
            data_name=data_name,
            train_indices=None,
            valid_indices=None,
            eval_batch_size=16,
        )

    model = construct_model(data_name=data_name)
    if data_name == "rte":
        model.load_state_dict(
            torch.load(
                f"../files/checkpoints/data_{data_name}/model_{model_id}/iter_228.pt",
                map_location="cpu",
            )
        )
    else:
        model.load_state_dict(
            torch.load(
                f"../files/checkpoints/data_{data_name}/model_{model_id}/iter_4800.pt",
                map_location="cpu",
            )
        )
    model.eval()
    task = GlueExperimentTask(device=DEVICE)
    return model.to(device=DEVICE), eval_train_loader, valid_loader, task


def get_single_trajectory_checkpoints(
    data_name: str,
    model_id: int = 0,
) -> List[str]:
    if data_name == "rte":
        iter_list = [38, 76, 114, 152, 190, 228]
    else:
        iter_list = [800, 1600, 2400, 3200, 4000, 4800]
    assert len(iter_list) == 6

    checkpoints = []
    for n_iter in iter_list:
        checkpoints.append(
            f"../files/checkpoints/data_{data_name}/model_{model_id}/iter_{n_iter}.pt"
        )
    return checkpoints


def compute_all_baselines(data_name: str, model_id: int) -> None:
    hyper_dict = get_hyperparameters(data_name)
    lrs = float(hyper_dict["lr"])  # This pipeline uses a fixed learning rate.
    ln_task = GlueWithLayerNormExperimentTask(device=DEVICE)

    expt_name = "representation_similarity_dot"
    file_name = get_default_file_name(
        base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, model_id=model_id
    )
    if os.path.exists(file_name):
        print(f"Found existing scores at {file_name}.")
    else:
        model, eval_train_loader, valid_loader, _ = prepare_everything(
            data_name=data_name,
            model_id=model_id,
        )
        scores = compute_representation_similarity(
            model=model,
            metric="dot",
            eval_train_loader=eval_train_loader,
            valid_loader=valid_loader,
            task=ln_task,
        )
        save_tensor(tensor=scores, file_name=file_name, overwrite=False)

    expt_name = "gradient_similarity_dot"
    file_name = get_default_file_name(
        base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, model_id=model_id
    )
    if os.path.exists(file_name):
        print(f"Found existing results at {file_name}.")
    else:
        model, eval_train_loader, valid_loader, _ = prepare_everything(
            data_name=data_name,
            model_id=model_id,
        )
        scores = compute_gradient_similarity(
            model=model,
            metric="dot",
            eval_train_loader=eval_train_loader,
            valid_loader=valid_loader,
            task=ln_task,
        )
        save_tensor(tensor=scores, file_name=file_name, overwrite=False)

    expt_name = "tracin_dot"
    file_name = get_default_file_name(
        base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, model_id=model_id
    )
    if os.path.exists(file_name):
        print(f"Found existing results at {file_name}.")
    else:
        model, eval_train_loader, valid_loader, _ = prepare_everything(
            data_name=data_name,
            model_id=model_id,
        )
        checkpoints = get_single_trajectory_checkpoints(
            data_name=data_name,
            model_id=model_id,
        )
        scores = compute_tracin(
            model=model,
            metric="dot",
            checkpoints=checkpoints,
            lrs=lrs,
            eval_train_loader=eval_train_loader,
            valid_loader=valid_loader,
            task=ln_task,
        )
        save_tensor(tensor=scores, file_name=file_name, overwrite=False)
        del model, eval_train_loader, valid_loader

    expt_name = "tracin_cos"
    file_name = get_default_file_name(
        base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, model_id=model_id
    )
    if os.path.exists(file_name):
        print(f"Found existing results at {file_name}.")
    else:
        model, eval_train_loader, valid_loader, _ = prepare_everything(
            data_name=data_name,
            model_id=model_id,
        )
        checkpoints = get_single_trajectory_checkpoints(
            data_name=data_name,
            model_id=model_id,
        )
        scores = compute_tracin(
            model=model,
            metric="cos",
            checkpoints=checkpoints,
            lrs=lrs,
            eval_train_loader=eval_train_loader,
            valid_loader=valid_loader,
            task=ln_task,
        )
        save_tensor(tensor=scores, file_name=file_name, overwrite=False)


def compute_all_trak(data_name: str, model_id: int) -> None:
    expt_name = "trak"
    file_name = get_default_file_name(
        base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, model_id=model_id
    )
    if os.path.exists(file_name):
        print(f"Found existing scores at {file_name}.")
    else:
        model, eval_train_loader, valid_loader, task = prepare_everything(
            data_name=data_name,
            model_id=model_id,
        )
        checkpoints = get_single_trajectory_checkpoints(
            data_name=data_name,
            model_id=model_id,
        )
        # Use last 3 checkpoints.
        checkpoints = checkpoints[-3:]
        assert len(checkpoints) == 3
        print(checkpoints)

        scores = compute_trak(
            model=model,
            expt_name=f"{data_name}_{model_id}_trak_speed",
            proj_dim=4096,
            checkpoints=checkpoints,
            train_loader=None,
            eval_train_loader=eval_train_loader,
            valid_loader=valid_loader,
            task=task,
            use_half_precision=False,
        )
        save_tensor(tensor=scores, file_name=file_name, overwrite=False)


if __name__ == "__main__":
    compute_all_trak(data_name="qnli", model_id=0)

