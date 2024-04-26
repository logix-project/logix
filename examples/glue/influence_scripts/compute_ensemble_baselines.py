import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from experiments.compute_utils import (
    compute_influence_function,
    compute_representation_similarity,
    compute_tracin,
    get_ensemble_file_name, compute_trak,
)
from experiments.glue.pipeline import construct_model, get_hyperparameters, get_loaders
from experiments.glue.task import GlueExperimentTask, GlueWithLayerNormExperimentTask
from experiments.utils import save_tensor
from src.abstract_task import AbstractTask

BASE_PATH = "../files/ensemble_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


def prepare_everything(
    data_name: str, alpha: str, model_id: int = 0
) -> Tuple[
    nn.Module, torch.utils.data.DataLoader, torch.utils.data.DataLoader, AbstractTask
]:
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(f"{BASE_PATH}/data_{data_name}/", exist_ok=True)
    os.makedirs(f"{BASE_PATH}/data_{data_name}/alpha_{alpha}/", exist_ok=True)
    if data_name == "rte":
        _, eval_train_loader, valid_loader = get_loaders(
            data_name=data_name,
            train_indices=None,
            valid_indices=None,
            eval_batch_size=16,
        )
    else:
        _, eval_train_loader, valid_loader = get_loaders(
            data_name=data_name,
            train_indices=None,
            valid_indices=list(range(512)),
            eval_batch_size=16,
        )
    model = construct_model(data_name=data_name)
    if data_name == "rte":
        model.load_state_dict(
            torch.load(
                f"../files/subset_checkpoints/data_{data_name}/alpha_{alpha}/model_{model_id}/iter_228.pt",
                map_location="cpu",
            )
        )
    else:
        model.load_state_dict(
            torch.load(
                f"../files/subset_checkpoints/data_{data_name}/alpha_{alpha}/model_{model_id}/iter_4800.pt",
                map_location="cpu",
            )
        )
    model.eval()
    task = GlueExperimentTask(device=DEVICE)
    return model.to(device=DEVICE), eval_train_loader, valid_loader, task


def get_single_trajectory_checkpoints(
    data_name: str,
    alpha: str,
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
            f"../files/subset_checkpoints/data_{data_name}/alpha_{alpha}/model_{model_id}/iter_{n_iter}.pt"
        )
    return checkpoints


def get_alpha_checkpoints(
    data_name: str,
    alpha: str,
) -> List[str]:
    checkpoints = []
    for mid in range(10):
        if data_name == "rte":
            checkpoints.append(
                f"../files/subset_checkpoints/data_{data_name}/alpha_{alpha}/model_{mid}/iter_228.pt"
            )
        else:
            checkpoints.append(
                f"../files/subset_checkpoints/data_{data_name}/alpha_{alpha}/model_{mid}/iter_4800.pt"
            )
    return checkpoints


def compute_all_baselines(
    data_name: str, model_ids: List[int], alphas: List[str]
) -> None:
    hyper_dict = get_hyperparameters(data_name)
    lrs = float(hyper_dict["lr"])  # This pipeline uses a fixed learning rate.
    ln_task = GlueWithLayerNormExperimentTask(device=DEVICE)

    for alpha in alphas:
        # expt_name = "representation_similarity_dot"
        # file_name = get_ensemble_file_name(
        #     base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, alpha=alpha
        # )
        # if os.path.exists(file_name):
        #     print(f"Found existing results at {file_name}.")
        # else:
        #     score_table = 0.0
        #     for mid in model_ids:
        #         model, eval_train_loader, valid_loader, _ = prepare_everything(
        #             data_name=data_name, model_id=mid, alpha=alpha
        #         )
        #         scores = compute_representation_similarity(
        #             model=model,
        #             metric="cos",
        #             eval_train_loader=eval_train_loader,
        #             valid_loader=valid_loader,
        #             task=ln_task,
        #         )
        #         score_table += scores
        #         del model, eval_train_loader, valid_loader, scores
        #     score_table /= len(model_ids)
        #     save_tensor(tensor=score_table, file_name=file_name, overwrite=False)

        # expt_name = "gradient_similarity_dot"
        # file_name = get_ensemble_file_name(
        #     base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, alpha=alpha
        # )
        # if os.path.exists(file_name):
        #     print(f"Found existing results at {file_name}.")
        # else:
        #     score_table = 0.0
        #     for mid in model_ids:
        #         model, eval_train_loader, valid_loader, _ = prepare_everything(
        #             data_name=data_name, model_id=mid, alpha=alpha
        #         )
        #         scores = compute_gradient_similarity(
        #             model=model,
        #             metric="dot",
        #             eval_train_loader=eval_train_loader,
        #             valid_loader=valid_loader,
        #             task=ln_task,
        #         )
        #         score_table += scores
        #         del model, eval_train_loader, valid_loader, scores
        #     score_table /= len(model_ids)
        #     save_tensor(tensor=score_table, file_name=file_name, overwrite=False)

        # expt_name = "tracin_dot"
        # file_name = get_ensemble_file_name(
        #     base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, alpha=alpha
        # )
        # if os.path.exists(file_name):
        #     print(f"Found existing results at {file_name}.")
        # else:
        #     score_table = 0.0
        #     for mid in model_ids:
        #         model, eval_train_loader, valid_loader, _ = prepare_everything(
        #             data_name=data_name, model_id=mid, alpha=alpha
        #         )
        #         checkpoints = get_single_trajectory_checkpoints(
        #             data_name=data_name, model_id=mid, alpha=alpha
        #         )
        #         scores = compute_tracin(
        #             model=model,
        #             metric="dot",
        #             checkpoints=checkpoints,
        #             lrs=lrs,
        #             eval_train_loader=eval_train_loader,
        #             valid_loader=valid_loader,
        #             task=ln_task,
        #         )
        #         score_table += scores
        #         del model, eval_train_loader, valid_loader, scores
        #     score_table /= len(model_ids)
        #     save_tensor(tensor=score_table, file_name=file_name, overwrite=False)

        # expt_name = "tracin_cos"
        # file_name = get_ensemble_file_name(
        #     base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, alpha=alpha
        # )
        # if os.path.exists(file_name):
        #     print(f"Found existing results at {file_name}.")
        # else:
        #     score_table = 0.0
        #     for mid in model_ids:
        #         model, eval_train_loader, valid_loader, _ = prepare_everything(
        #             data_name=data_name, model_id=mid, alpha=alpha
        #         )
        #         checkpoints = get_single_trajectory_checkpoints(
        #             data_name=data_name, model_id=mid, alpha=alpha
        #         )
        #         scores = compute_tracin(
        #             model=model,
        #             metric="cos",
        #             checkpoints=checkpoints,
        #             lrs=lrs,
        #             eval_train_loader=eval_train_loader,
        #             valid_loader=valid_loader,
        #             task=ln_task,
        #         )
        #         score_table += scores
        #         del model, eval_train_loader, valid_loader, scores
        #     score_table /= len(model_ids)
        #     save_tensor(tensor=score_table, file_name=file_name, overwrite=False)

        expt_name = "trak"
        file_name = get_ensemble_file_name(
            base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, alpha=alpha
        )
        if os.path.exists(file_name):
            print(f"Found existing results at {file_name}.")
        else:
            model, eval_train_loader, valid_loader, task = prepare_everything(
                data_name=data_name, model_id=0, alpha=alpha
            )
            checkpoints = get_alpha_checkpoints(data_name=data_name, alpha=alpha)
            print(checkpoints)
            scores = compute_trak(
                model=model,
                expt_name=f"{data_name}_{alpha}_trak_ensemble_model",
                proj_dim=4096,
                checkpoints=checkpoints,
                train_loader=None,
                eval_train_loader=eval_train_loader,
                valid_loader=valid_loader,
                task=task,
            )
            save_tensor(tensor=scores, file_name=file_name, overwrite=False)


def compute_if(
    data_name: str,
    model_ids: List[int],
    alphas: List[str],
    n_epoch: int = 1,
    damping: Optional[float] = 1e-08,
    overwrite: bool = False,
) -> None:
    for alpha in alphas:
        if damping is None:
            damp_name = "heuristic"
        else:
            damp_name = damping
        expt_name = f"if_d{damp_name}"
        file_name = get_ensemble_file_name(
            base_path=BASE_PATH, expt_name=expt_name, data_name=data_name, alpha=alpha
        )
        if os.path.exists(file_name) and not overwrite:
            print(f"Found existing results at {file_name}.")
        else:
            score_table = 0.0
            for mid in model_ids:
                model, eval_train_loader, valid_loader, task = prepare_everything(
                    data_name=data_name, model_id=mid, alpha=alpha
                )
                scores = compute_influence_function(
                    model=model,
                    n_epoch=n_epoch,
                    task=task,
                    damping=damping,
                    train_loader=eval_train_loader,
                    eval_train_loader=eval_train_loader,
                    valid_loader=valid_loader,
                )
                score_table += scores
                del model, eval_train_loader, valid_loader, task, scores
            score_table /= len(model_ids)
            save_tensor(tensor=score_table, file_name=file_name, overwrite=overwrite)


if __name__ == "__main__":
    compute_all_baselines(
        data_name="qnli", model_ids=list(range(10)), alphas=["0.5"]
    )
    # import sys
    #
    # try:
    #     option = int(sys.argv[1])
    #     if option == 1:
    #         compute_all_baselines(
    #             data_name="qnli", model_ids=list(range(10)), alphas=["0.0"]
    #         )
    #         compute_all_baselines(
    #             data_name="sst2", model_ids=list(range(10)), alphas=["0.0"]
    #         )
    #     elif option == 2:
    #         compute_if(data_name="qnli", model_ids=list(range(10)), alphas=["0.0"])
    #         compute_if(data_name="sst2", model_ids=list(range(10)), alphas=["0.0"])
    #     elif option == 3:
    #         compute_all_baselines(
    #             data_name="rte", model_ids=list(range(10)), alphas=["0.0"]
    #         )
    #         compute_if(data_name="rte", model_ids=list(range(10)), alphas=["0.0"])
    #     else:
    #         raise NotImplementedError(f"Not an available option {option}.")
    # except IndexError:
    #     print("Select the right option!")
