import os
from typing import List, Optional, Tuple
import time
import torch
import torch.nn as nn
from trak import TRAKer

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


def process_batch(batch):
    try:
        return (
            batch["input_ids"],
            batch["token_type_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
    except:
        return (
            batch["pixel_values"],
            batch["labels"],
        )

def prepare_everything(
    data_name: str,
    model_id: int = 0,
) -> Tuple[
    nn.Module, torch.utils.data.DataLoader, torch.utils.data.DataLoader, AbstractTask
]:
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(f"{BASE_PATH}/data_{data_name}/", exist_ok=True)
    os.makedirs(f"{BASE_PATH}/data_{data_name}/model_{model_id}/", exist_ok=True)

    _, eval_train_loader, _ = get_loaders(
        data_name=data_name,
        train_indices=None,
        valid_indices=list(range(2000)),
        eval_batch_size=32,
    )
    _, _, valid_loader = get_loaders(
        data_name=data_name,
        train_indices=None,
        valid_indices=list(range(512)),
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


def compute_all_trak(data_name: str, model_id: int) -> None:
    expt_name = "trak"
    file_name = get_default_file_name(
        base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, model_id=model_id
    )

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

    traker = TRAKer(
        model=model,
        load_from_save_dir=True,
        task=task.get_model_output(),
        proj_dim=4096,
        train_set_size=len(eval_train_loader.dataset),
        device=DEVICE,
        save_dir=f"speed_5checkpoints",
        use_half_precision=False,
        proj_max_batch_size=32,  # Could change?
    )

    print("Start Time")
    start_time = time.time()
    print(start_time)

    for model_id, ckpt in enumerate(checkpoints):
        traker.load_checkpoint(
            torch.load(ckpt, map_location=task.device), model_id=model_id
        )
        for batch in eval_train_loader:
            if isinstance(batch, dict):
                batch = process_batch(batch)
            batch = [x.to(task.device) for x in batch]

            traker.featurize(
                batch=batch, num_samples=task.get_batch_size(batch)
            )
        traker.finalize_features()

    for model_id, ckpt in enumerate(checkpoints):
        traker.start_scoring_checkpoint(
            exp_name="speed5",
            checkpoint=torch.load(ckpt, map_location=task.device),
            model_id=model_id,
            num_targets=len(valid_loader.dataset),
        )
        for batch in valid_loader:
            if isinstance(batch, dict):
                batch = process_batch(batch)
            batch = [x.to(task.device) for x in batch]
            traker.score(batch=batch, num_samples=task.get_batch_size(batch))
    scores = traker.finalize_scores(exp_name="speed5")
    print("End Time")
    end_time = time.time()
    print(end_time)

    print("Total Took")
    print(end_time - start_time)


if __name__ == "__main__":
    compute_all_trak(data_name="qnli", model_id=0)

