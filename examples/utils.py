import gc
import os
import random
import struct
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def reset_seed() -> None:
    rng_seed = struct.unpack("I", os.urandom(4))[0]
    set_seed(rng_seed)


def clear_gpu_cache() -> None:
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def save_tensor(tensor: Any, file_name: str, overwrite: bool = False) -> bool:
    if not os.path.exists(file_name):
        torch.save(tensor, file_name)
        return False
    else:
        if overwrite:
            print(f"Removing existing file at {file_name}.")
            torch.save(tensor, file_name)
        else:
            print("Found existing file. Skipping.")
        return True
