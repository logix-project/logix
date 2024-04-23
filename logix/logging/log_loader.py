from collections import OrderedDict
from functools import reduce

import numpy as np
import torch
from torch.utils.data import Dataset

from logix.logging.log_loader_utils import (
    get_entry_metadata,
    get_flatten_item,
    get_mmap_data,
    get_mmap_metadata,
    find_chunk_indices,
)


class LogDataset(Dataset):
    def __init__(self, log_dir, flatten):
        self.chunk_indices = None
        self.memmaps = []

        self.data_id_to_chunk = OrderedDict()
        self.log_dir = log_dir
        self.flatten = flatten

        # Find all chunk indices
        self.chunk_indices = find_chunk_indices(self.log_dir)
        self.fetch_data()
        self.data_id_list = list(self.data_id_to_chunk.keys())

    def fetch_data(self):
        # Add metadata and mmap files for all indices.
        for idx, chunk_index in enumerate(self.chunk_indices):
            file_root = f"log_{chunk_index}"
            mmap_filename = f"{file_root}.mmap"
            entry = get_mmap_data(self.log_dir, mmap_filename)
            self.memmaps.append(entry)

            self.data_id_to_chunk = get_mmap_metadata(
                self.data_id_to_chunk,
                self.log_dir,
                f"{file_root}_metadata.json",
                idx,
            )

    def __getitem__(self, index):
        data_id = self.data_id_list[index]
        chunk_idx, entry = self.data_id_to_chunk[data_id]
        nested_dict = {}
        mmap = self.memmaps[chunk_idx]
        offset = entry["offset"]
        if self.flatten:
            return data_id, get_flatten_item(
                mmap, offset, entry["block_size"], entry["dtype"]
            )
        dtype = entry["dtype"]
        for i in range(len(entry["path"])):
            path = entry["path"][i]
            shape = tuple(entry["shape"][i])
            tensor = torch.from_numpy(
                np.ndarray(shape, dtype, buffer=mmap, offset=offset, order="C")
            ).clone()

            current_level = nested_dict
            for key in path[:-1]:
                if key not in current_level:
                    current_level[key] = {}
                current_level = current_level[key]
            current_level[path[-1]] = tensor
            offset += reduce(lambda x, y: x * y, shape) * np.dtype(dtype).itemsize

        assert (
            offset == entry["offset"] + entry["block_size"] * np.dtype(dtype).itemsize
        ), f"the block_size does not match the shape for data_id: {entry['data_id']}"
        return data_id, nested_dict

    def __len__(self):
        return len(self.data_id_to_chunk)

    def close(self):
        for mmap in self.memmaps:
            del mmap
