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
from src.gradient_similarity import GradientSimilarityComputer
from src.influence_function import InfluenceFunctionComputer
from src.tracin import TracinComputer

BASE_PATH = "../files/corruption_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_everything(
    data_name: str,
    model_id: int = 0,
) -> Tuple[
    nn.Module, torch.utils.data.DataLoader, torch.utils.data.DataLoader, AbstractTask
]:
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(f"{BASE_PATH}/data_{data_name}/", exist_ok=True)
    os.makedirs(f"{BASE_PATH}/data_{data_name}/model_{model_id}/", exist_ok=True)

    _, eval_train_loader, valid_loader = get_loaders(
        data_name=data_name,
        train_indices=None,
        valid_indices=list(range(512)),
        eval_batch_size=16,
    )

    model = construct_model(data_name=data_name)
    model.load_state_dict(
        torch.load(
            f"../files/checkpoints/data_{data_name}_corrupted/model_{model_id}/iter_4800.pt",
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
    iter_list = [800, 1600, 2400, 3200, 4000, 4800]
    assert len(iter_list) == 6

    checkpoints = []
    for n_iter in iter_list:
        checkpoints.append(
            f"../files/checkpoints/data_{data_name}_corrupted/model_{model_id}/iter_{n_iter}.pt"
        )
    return checkpoints


def compute_all_baselines(data_name: str, model_id: int) -> None:
    hyper_dict = get_hyperparameters(data_name)
    lrs = float(hyper_dict["lr"])  # This pipeline uses a fixed learning rate.
    ln_task = GlueWithLayerNormExperimentTask(device=DEVICE)

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
        computer = GradientSimilarityComputer(
            model=model,
            task=ln_task,
            metric="dot",
        )
        scores = computer.compute_self_scores_with_loader(loader=eval_train_loader)
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
        computer = TracinComputer(
            model=model,
            task=ln_task,
            metric="dot",
        )
        scores = computer.compute_self_scores_with_loader(
            checkpoints=checkpoints,
            lrs=lrs,
            loader=eval_train_loader,
        )
        save_tensor(tensor=scores, file_name=file_name, overwrite=False)


def compute_if(
    data_name: str,
    model_id: int,
    n_epoch: int = 1,
    damping: Optional[float] = 1e-08,
    overwrite: bool = False,
) -> None:
    if damping is None:
        damp_name = "heuristic"
    else:
        damp_name = damping

    expt_name = f"if_d{damp_name}"
    file_name = get_default_file_name(
        base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, model_id=model_id
    )
    if os.path.exists(file_name) and not overwrite:
        print(f"Found existing results at {file_name}.")
    else:
        model, eval_train_loader, valid_loader, task = prepare_everything(
            data_name=data_name,
            model_id=model_id,
        )
        computer = InfluenceFunctionComputer(
            model=model,
            task=task,
            n_epoch=n_epoch,
            damping=damping,
        )
        computer.build_curvature_blocks(loader=eval_train_loader)
        scores = computer.compute_self_scores_with_loader(loader=eval_train_loader)
        save_tensor(tensor=scores, file_name=file_name, overwrite=overwrite)


# def compute_unif(
#     model_id: int,
#     n_epoch: int = 1,
#     overwrite: bool = False,
# ) -> None:
#     data_name = "cifar10"
#     n_iter = 50_000 * 25
#     lr = 0.2
#
#     checkpoints = get_single_trajectory_checkpoints(
#         model_id=model_id,
#     )
#     checkpoints.reverse()
#
#     expt_name = "unif_average"
#     file_name = get_default_file_name(
#         base_path=BASE_PATH,
#         expt_name=expt_name,
#         data_name=data_name,
#         model_id=model_id,
#     )
#     if os.path.exists(file_name) and not overwrite:
#         print(f"Found existing results at {file_name}.")
#     else:
#         (
#             model,
#             eval_train_loader,
#             eval_train_loader_with_aug,
#             _,
#             task,
#         ) = prepare_everything(
#             model_id=model_id,
#         )
#         computer = UnifAverageComputer(
#             model=model,
#             task=task,
#             n_epoch=n_epoch,
#             n_iters=n_iter,
#             lrs=lr,
#             checkpoints=checkpoints,
#         )
#         computer.build_curvature_blocks(
#             loader=eval_train_loader_with_aug,
#         )
#         scores = computer.compute_self_scores_with_loader(
#             loader=eval_train_loader,
#         )
#         save_tensor(tensor=scores, file_name=file_name, overwrite=overwrite)
#
#     checkpoints = get_single_trajectory_checkpoints(
#         model_id=model_id,
#     )
#     checkpoints.reverse()
#     if data_name == "cifar2":
#         chunk1 = checkpoints[:5]
#         chunk2 = checkpoints[4:8]
#         chunk3 = checkpoints[7:]
#         chunk_lst = [chunk1, chunk2, chunk3]
#         iter_lst = [40 * 10_000, 30 * 10_000, 30 * 10_000]
#         lr_lst = [0.398, 0.261, 0.084]
#     else:
#         chunk1 = checkpoints[:4]
#         chunk2 = checkpoints[3:7]
#         chunk3 = checkpoints[6:]
#         chunk_lst = [chunk1, chunk2, chunk3]
#         iter_lst = [7 * 50_000, 9 * 50_000, 9 * 50_000]
#         lr_lst = [0.0701, 0.2301, 0.2709]
#
#     expt_name = "unif_segment"
#     file_name = get_default_file_name(
#         base_path=BASE_PATH,
#         expt_name=expt_name,
#         data_name=data_name,
#         model_id=model_id,
#     )
#     if os.path.exists(file_name) and not overwrite:
#         print(f"Found existing results at {file_name}.")
#     else:
#         (
#             model,
#             eval_train_loader,
#             eval_train_loader_with_aug,
#             _,
#             task,
#         ) = prepare_everything(
#             model_id=model_id,
#         )
#         computer = UnifSegmentComputer(
#             model=model,
#             task=task,
#             n_epoch=n_epoch,
#             n_iters=iter_lst,
#             lrs=lr_lst,
#             checkpoints=chunk_lst,
#         )
#         computer.build_curvature_blocks(
#             loader=eval_train_loader_with_aug,
#         )
#         scores = computer.compute_self_scores_with_loader(
#             loader=eval_train_loader,
#         )
#         save_tensor(tensor=scores, file_name=file_name, overwrite=overwrite)


if __name__ == "__main__":
    data_names = ["qnli", "sst2"]
    for dn in data_names:
        for mid in range(5):
            compute_all_baselines(data_name=dn, model_id=mid)
            compute_if(data_name=dn, model_id=mid)
            # compute_unif(model_id=mid, overwrite=True)
