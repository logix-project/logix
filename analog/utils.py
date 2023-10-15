from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    else:
        raise ValueError("Unsupported tensor type. Supported libraries: NumPy, PyTorch")


def get_world_size(group=None):
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
