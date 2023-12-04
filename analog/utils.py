import sys
import logging
from typing import Any, List
from collections import defaultdict

import hashlib

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange


_logger = None


def get_logger() -> logging.Logger:
    """
    Get global logger.
    """
    global _logger
    if _logger:
        return _logger
    logger = logging.getLogger("AnaLog")
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    logger.propagate = False
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    _logger = logger
    return _logger


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    else:
        raise ValueError("Unsupported tensor type. Supported libraries: NumPy, PyTorch")


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 0


def get_rank(group=None) -> int:
    if dist.is_initialized():
        return dist.get_rank(group)
    else:
        return 0


def nested_dict():
    """
    Helper function to create a nested defaultdict.
    """
    return defaultdict(nested_dict)


def deep_get(d, keys):
    if not keys or d is None:
        return d
    return deep_get(d.get(keys[0]), keys[1:])


class DataIDGenerator:
    def __init__(self, mode="hash") -> None:
        self.mode = mode
        if mode == "index":
            self.count = 0

    def __call__(self, data: Any):
        if self.mode == "hash":
            return self.generate_hash_id(data)
        elif self.mode == "index":
            return self.generate_index_id(data)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def generate_hash_id(self, data: Any) -> List[str]:
        data_id = []
        for d in data:
            ndarray = to_numpy(d)
            ndarray.flags.writeable = False
            data_id.append(hashlib.sha256(ndarray.tobytes()).hexdigest())
        return data_id

    def generate_index_id(self, data: Any) -> List[int]:
        data_id = np.arange(self.count, self.count + len(data))
        data_id = [str(d) for d in data_id]
        self.count += len(data)
        return data_id


def stack_tensor(tensor_list):
    return rearrange(tensor_list, "a ... -> a ...")
