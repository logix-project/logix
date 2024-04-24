import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision


def construct_model(name) -> nn.Module:
    if name == "mnist" or name == "fmnist":
        from examples.mnist.pipeline import construct_model
        model = construct_model()
    elif name == "cifar10":
        from examples.cifar.pipeline import construct_model
        model = construct_model(name)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return model


def get_hyperparameters(data_name: str) -> Dict[str, float]:
    # See `scripts/find_hyperparameters.py` to examine how these values were selected.
    if "mnist" in data_name:
        from examples.mnist.pipeline import get_hyperparameters
    elif "cifar10" in data_name:
        from examples.cifar.pipeline import get_hyperparameters
    else:
        raise ValueError(f"Unknown dataset: {data_name}")
    return get_hyperparameters(data_name)

def get_remove_intervals(data_name: str) -> List[int]:
    if data_name == "mnist" or data_name == "fmnist":
        return [50, 100, 150, 200, 250, 300]
    elif data_name == "cifar10":
        return [200, 400, 600, 800, 1000, 1200]

def get_loaders(
    data_name: str,
    eval_batch_size: int = 1024,
    train_indices: Optional[List[int]] = None,
    valid_indices: Optional[List[int]] = None,
    do_corrupt: bool = False,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    if "mnist" in data_name:
        from examples.mnist.pipeline import get_loaders
    elif "cifar10" in data_name:
        from examples.cifar.pipeline import get_loaders
    else:
        raise ValueError(f"Unknown dataset: {data_name}")
    return get_loaders(data_name, eval_batch_size, train_indices, valid_indices, do_corrupt)

def get_eval_train_loader_with_aug(
    data_name: str, eval_batch_size: int = 1024, train_indices: Optional[List[int]] = None, do_corrupt: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if "cifar10" in data_name or "cifar2" in data_name:
        from examples.cifar.pipeline import get_eval_train_loader_with_aug
    else:
        raise ValueError(f"Unknown dataset: {data_name}")
    return get_eval_train_loader_with_aug(data_name, eval_batch_size, train_indices, do_corrupt)
