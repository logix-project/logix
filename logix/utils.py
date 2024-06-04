# Copyright 2023-present the LogIX team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import logging as default_logging
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

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


def get_repr_dim(named_modules) -> Tuple[List[str], List[int]]:
    repr_dims = []
    paths = []
    for k, v in named_modules.items():
        get_logger().info(f"{v}: {k}")
        repr_dims.append(k.weight.data.numel())
        paths.append((v, "grad"))  # hardcoded
    return paths, repr_dims


def print_tracked_modules(repr_dim) -> None:
    """
    Print the tracked modules.
    """
    get_logger().info("Tracking the following modules:")
    get_logger().info(f"Total number of parameters: {repr_dim:,}\n")


def module_check(
    module: nn.Module,
    module_name: str,
    supported_modules: Optional[List[nn.Module]] = None,
    type_filter: Optional[List[nn.Module]] = None,
    name_filter: Optional[List[str]] = None,
    is_lora: bool = False,
) -> bool:
    """
    Check if the module is supported for logging.

    Args:
        module (nn.Module): The module to check.
        module_name (str): Name of the module.
        supported_modules (Optional[List[nn.Module]]): List of supported module types.
        type_filter (Optional[List[nn.Module]]): List of module types to filter.
        name_filter (Optional[List[str]]): List of keywords to filter module names.
        is_lora (bool): Flag to check for specific 'analog_lora_B' in module names.

    Returns:
        bool: True if module is supported, False otherwise.
    """
    if list(module.children()):
        return False
    if supported_modules and not isinstance(module, tuple(supported_modules)):
        return False
    if type_filter and not isinstance(module, tuple(type_filter)):
        return False
    if name_filter and not any(keyword in module_name for keyword in name_filter):
        return False
    if is_lora and "logix_lora_B" not in module_name:
        return False
    return True


def nested_dict():
    """
    Helper function to create a nested defaultdict.
    """
    return defaultdict(nested_dict)


def merge_log_dict(merged_log_dict, log_dict) -> None:
    for key, value in log_dict.items():
        if isinstance(value, dict):
            merge_log_dict(merged_log_dict[key], value)
        else:
            if key not in merged_log_dict:
                merged_log_dict[key] = value
            else:
                merged_log_dict[key] = torch.cat([merged_log_dict[key], value], dim=0)


def merge_logs(log_list):
    merged_data_id = []
    merged_log_dict = nested_dict()

    for data_id, log_dict in log_list:
        merged_data_id.extend(data_id)
        merge_log_dict(merged_log_dict, log_dict)
    return merged_data_id, merged_log_dict


def flatten_log(log, path) -> torch.Tensor:
    flat_log_list = []
    for module, log_type in path:
        log_module = log[module][log_type]
        bsz = log_module.shape[0]
        flat_log_list.append(log_module.reshape(bsz, -1))
    flat_log = torch.cat(flat_log_list, dim=1)

    return flat_log


def unflatten_log(log, path):
    raise NotImplementedError


def synchronize_device(
    src: Dict[str, Dict[str, torch.Tensor]],
    tgt: Dict[str, Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
) -> None:
    """
    Synchronize the device of two tensor dicts.

    Args:
        src (Dict[str, Dict[str, torch.Tensor]]): Source tensors
        tgt (Dict[str, Dict[str, torch.Tensor]]): Target tensors
        device (Optional[torch.device]): Device to synchronize to
    """

    for module_name, module_dict in tgt.items():
        for log in module_dict.keys():
            if device is None:
                device = src[module_name][log].device
            tgt[module_name][log] = tgt[module_name][log].to(device=device)


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

    def generate_index_id(self, data: Any) -> List[str]:
        """
        Given data, generate id based on the index.
        """
        data_id = np.arange(self.count, self.count + len(data))
        data_id = [str(d) for d in data_id]
        self.count += len(data)
        return data_id
