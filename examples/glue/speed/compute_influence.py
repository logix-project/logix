import os
import time
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
from src.influence_function import InfluenceFunctionComputer

BASE_PATH = "../files/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


EVAL_BATCH_SIZE = 32
VALID_BATCH_SIZE = 128


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
        eval_batch_size=EVAL_BATCH_SIZE,
    )
    _, valid_loader, _ = get_loaders(
        data_name=data_name,
        # train_indices=list(range(2000)),
        valid_indices=list(range(2000)),
        eval_batch_size=VALID_BATCH_SIZE,
    )

    model = construct_model(data_name=data_name)
    # model.load_state_dict(
    #     torch.load(
    #         f"../files/checkpoints/data_{data_name}/model_{model_id}/iter_4800.pt",
    #         map_location="cpu",
    #     )
    # )
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


def compute_if(
    data_name: str,
    model_id: int,
    n_epoch: int = 1,
    damping: Optional[float] = 1e-08,
    overwrite: bool = False,
) -> None:

    model, eval_train_loader, valid_loader, task = prepare_everything(
        data_name=data_name,
        model_id=model_id,
    )

    embed = model.model.bert.embeddings.word_embeddings.weight
    delattr(model.model.bert.embeddings.word_embeddings, "weight")
    model.model.bert.embeddings.word_embeddings.register_buffer("weight", embed)

    embed = model.model.bert.embeddings.token_type_embeddings.weight
    delattr(model.model.bert.embeddings.token_type_embeddings, "weight")
    model.model.bert.embeddings.token_type_embeddings.register_buffer("weight", embed)

    embed = model.model.bert.embeddings.position_embeddings.weight
    delattr(model.model.bert.embeddings.position_embeddings, "weight")
    model.model.bert.embeddings.position_embeddings.register_buffer("weight", embed)

    # for module in model.modules():
    #     for name, param in module.named_parameters(recurse=False):
    #         if "embeddings" in name:
    #             delattr(module, name)  # Unregister parameter
    #             module.register_buffer(name, param)
    #         print(name)
    #     # delattr(module, name)  # Unregister parameter
    #     # module.register_buffer(name, param)

    # list(model.named_parameters())[0][1].requires_grad = False

    print("START")
    computer = InfluenceFunctionComputer(
        model=model, task=task, n_epoch=n_epoch, damping=damping,
    )
    _, eval_train_loader_raw, _ = get_loaders(
        data_name=data_name,
        train_indices=None,
        eval_batch_size=512,
    )
    _, eval_train_loader_raw2, _ = get_loaders(
        data_name=data_name,
        train_indices=None,
        eval_batch_size=64,
    )

    _, eval_train_loader, _ = get_loaders(
        data_name=data_name,
        train_indices=None,
        eval_batch_size=64,
    )
    _, _, valid_loader = get_loaders(
        data_name=data_name,
        # train_indices=list(range(2000)),
        valid_indices=list(range(2000)),
        eval_batch_size=64,
    )

    start_time = time.time()
    computer.fit_covariances(loader=eval_train_loader_raw)
    computer.fit_eigendecompositions(keep_cache=False)
    for handle in computer._handles:
        handle.remove()

    end_time = time.time()
    print("Time to compute Eigendecomposition:" )
    print(end_time - start_time)
    start_time = time.time()

    computer._handles = []
    computer.fit_additional_factors(loader=eval_train_loader_raw2)
    scores = computer.compute_scores_with_loader_fast(
        test_loader=valid_loader, train_loader=eval_train_loader
    )
    end_time = time.time()
    print("total time:" )
    print(end_time - start_time)


if __name__ == "__main__":
    compute_if(data_name="qnli", model_id=0, damping=1e-08, overwrite=True)
