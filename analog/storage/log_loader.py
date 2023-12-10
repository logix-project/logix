from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset

from analog.storage.log_loader_util import (
    get_mmap_data,
    get_mmap_metadata,
    find_chunk_indices,
)


class DefaultLogDataset(Dataset):
    def __init__(self, log_dir):
        self.chunk_indices = None
        self.memmaps = []
        self.data_id_to_chunk = OrderedDict()
        self.log_dir = log_dir

        # Find all chunk indices
        self.chunk_indices = find_chunk_indices(self.log_dir)
        self.fetch_data()

    def fetch_data(self):
        # Add metadata and mmap files for all indices.
        for chunk_index in self.chunk_indices:
            file_root = f"log_chunk_{chunk_index}"
            mmap_filename = f"{file_root}.mmap"
            entry = get_mmap_data(self.log_dir, mmap_filename)
            print("e" + str(len(entry)))
            self.memmaps.append(entry)
            self.data_id_to_chunk = get_mmap_metadata(
                self.data_id_to_chunk,
                self.log_dir,
                f"{file_root}_metadata.json",
                chunk_index,
            )

    def __getitem__(self, index):
        data_id = list(self.data_id_to_chunk.keys())[index]
        chunk_idx, entries = self.data_id_to_chunk[data_id]
        nested_dict = {}
        mmap = self.memmaps[chunk_idx]

        for entry in entries:
            # Read the data and put it into the nested dictionary
            path = entry["path"]
            offset = entry["offset"]
            shape = tuple(entry["shape"])
            dtype = np.dtype(entry["dtype"])
            array = np.ndarray(shape, dtype, buffer=mmap, offset=offset, order="C")
            tensor = torch.Tensor(array)

            # Place the tensor in the correct location within the nested dictionary
            current_level = nested_dict
            for key in path[:-1]:
                if key not in current_level:
                    current_level[key] = {}
                current_level = current_level[key]
            current_level[path[-1]] = tensor
        return data_id, nested_dict

    def __len__(self):
        return len(self.data_id_to_chunk)

    def close(self):
        for mmap in self.memmaps:
            del mmap
