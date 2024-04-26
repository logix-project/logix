import os
from typing import List

import torch

from experiments.compute_utils import get_default_file_name
from experiments.glue.influence_scripts.compute_ensemble_baselines import (
    get_single_trajectory_checkpoints,
    prepare_everything,
)
from experiments.glue.pipeline import get_loaders
from experiments.utils import save_tensor
from src.unif.segment import UnifSegmentComputer

BASE_PATH = "../files/ensemble_unif_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


def get_single_trajectory_optimization_checkpoints(
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
            f"../files/subset_checkpoints/data_{data_name}/alpha_{alpha}/model_{model_id}/iter_{n_iter}_optimizer.pt"
        )
    return checkpoints


def compute_unif_segment(
    data_name: str,
    model_id: int,
    n_epoch: int = 1,
    overwrite: bool = False,
) -> None:
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(f"{BASE_PATH}/data_{data_name}/", exist_ok=True)
    os.makedirs(f"{BASE_PATH}/data_{data_name}/model_{model_id}/", exist_ok=True)

    if data_name == "sst2":
        lr = 3e-05
        iter_lst = [
            51_200 // 32,
            51_200 // 32,
            51_200 // 32,
        ]
    elif data_name == "rte":
        lr = 2e-05
        iter_lst = [
            2432 // 32,
            2432 // 32,
            2432 // 32,
        ]
    else:
        lr = 1e-05
        iter_lst = [
            51_200 // 32,
            51_200 // 32,
            51_200 // 32,
        ]

    checkpoints = get_single_trajectory_checkpoints(
        data_name=data_name,
        model_id=model_id,
        alpha="0.0",
    )
    checkpoints.reverse()
    chunk1 = checkpoints[:3]
    chunk2 = checkpoints[2:5]
    chunk3 = checkpoints[4:]
    chunk_lst = [chunk1, chunk2, chunk3]
    lr_lst = [lr, lr, lr]

    opt_checkpoints = get_single_trajectory_optimization_checkpoints(
        data_name=data_name,
        model_id=model_id,
        alpha="0.0",
    )
    opt_checkpoints.reverse()
    chunk1 = opt_checkpoints[:3]
    chunk2 = opt_checkpoints[2:5]
    chunk3 = opt_checkpoints[4:]
    opt_chunk_lst = [chunk1, chunk2, chunk3]

    expt_name = "unif_segment"
    file_name = get_default_file_name(
        base_path=BASE_PATH,
        expt_name=expt_name,
        data_name=data_name,
        model_id=model_id,
    )
    if os.path.exists(file_name) and not overwrite:
        print(f"Found existing results at {file_name}.")
    else:
        (
            train_loader,
            _,
            _,
        ) = get_loaders(
            data_name=data_name,
            train_indices=None,
        )
        model, eval_train_loader, valid_loader, task = prepare_everything(
            data_name=data_name,
            model_id=model_id,
            alpha="0.0",
        )
        checkpoints = get_single_trajectory_checkpoints(
            data_name=data_name,
            model_id=model_id,
            alpha="0.0"
        )
        checkpoints.reverse()

        print(checkpoints)
        print(opt_checkpoints)

        computer = UnifSegmentComputer(
            model=model,
            task=task,
            n_epoch=n_epoch,
            n_iters=iter_lst,
            lrs=lr_lst,
            checkpoints=chunk_lst,
            use_adam=True,
            optimizer_checkpoints=opt_chunk_lst,
        )
        computer.build_curvature_blocks(
            loader=eval_train_loader,
        )
        scores = computer.compute_scores_with_loader(
            test_loader=valid_loader,
            train_loader=eval_train_loader,
        )
        save_tensor(tensor=scores, file_name=file_name, overwrite=overwrite)


if __name__ == "__main__":
    import sys

    data_name = str(sys.argv[1])
    model_id = int(sys.argv[2])
    compute_unif_segment(data_name=data_name, model_id=model_id, overwrite=False)
