import sys
import logging as default_logging
from typing import Any, List
from collections import defaultdict

import hashlib

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange


_logger = None


class DistributedRankFilter(default_logging.Filter):
    """
    This is a logging filter which will filter out logs from all ranks
    in distributed training except for rank 0.
    """

    def filter(self, record):
        return get_rank() == 0


def get_logger() -> default_logging.Logger:
    """
    Get global logger.
    """
    global _logger
    if _logger:
        return _logger
    logger = default_logging.getLogger("AnaLog")
    log_format = default_logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    logger.propagate = False
    logger.setLevel(default_logging.INFO)
    ch = default_logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(default_logging.INFO)
    ch.setFormatter(log_format)

    # Apply the rank filter to the handler
    rank_filter = DistributedRankFilter()
    ch.addFilter(rank_filter)

    logger.addHandler(ch)

    _logger = logger
    return _logger


def to_numpy(tensor) -> np.ndarray:
    """
    Convert a tensor to NumPy array.

    Args:
        tensor: The tensor to be converted.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    else:
        raise ValueError("Unsupported tensor type. Supported libraries: NumPy, PyTorch")


def get_world_size() -> int:
    """
    Get the number of processes in the current distributed group.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 0


def get_rank(group=None) -> int:
    """
    Get the rank of the current process in the current distributed group.

    Args:
        group (optional): The process group to work on. If not specified,
            the default process group will be used.
    """
    if dist.is_initialized():
        return dist.get_rank(group)
    else:
        return 0


def print_tracked_modules(named_modules) -> None:
    """
    Print the tracked modules.
    """
    get_logger().info("Tracking the following modules:")
    repr_dim = 0
    for k, v in named_modules.items():
        get_logger().info(f"{v}: {k}")
        repr_dim += k.weight.data.numel()
    get_logger().info(f"Total number of parameters: {repr_dim:,}\n")


def nested_dict():
    """
    Helper function to create a nested defaultdict.
    """
    return defaultdict(nested_dict)


class DataIDGenerator:
    """
    Generate unique IDs for data.
    """

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
        """
        Given data, generate id using SHA256 hash.
        """
        data_id = []
        for d in data:
            ndarray = to_numpy(d)
            ndarray.flags.writeable = False
            data_id.append(hashlib.sha256(ndarray.tobytes()).hexdigest())
        return data_id

    def generate_index_id(self, data: Any) -> List[int]:
        """
        Given data, generate id based on the index.
        """
        data_id = np.arange(self.count, self.count + len(data))
        data_id = [str(d) for d in data_id]
        self.count += len(data)
        return data_id
