import os
from collections import OrderedDict

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, default_collate


class DefaultLogDataset(Dataset):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.schemas = []
        self.memmaps = []
        self.data_id_to_chunk = OrderedDict()

        # Find all chunk indices
        self.chunk_indices = self._find_chunk_indices(log_dir)

        # Add schemas and mmap files for all indices
        for chunk_index in self.chunk_indices:
            mmap_filename = os.path.join(log_dir, f"log_chunk_{chunk_index}.mmap")
            schema_filename = os.path.join(
                log_dir, f"log_chunk_{chunk_index}_metadata.json"
            )
            self._add_schema_and_mmap(schema_filename, mmap_filename, chunk_index)

    def _find_chunk_indices(self, directory):
        chunk_indices = []
        for filename in os.listdir(directory):
            if filename.endswith(".mmap"):
                parts = filename.rstrip(".mmap").split("_")
                if len(parts) != 0:
                    chunk_index = parts[-1]
                chunk_indices.append(int(chunk_index))
        return sorted(chunk_indices)

    def _add_schema_and_mmap(
        self, schema_filename, mmap_filename, chunk_index, dtype="uint8"
    ):
        # Load the schema
        with open(schema_filename, "r") as f:
            schema = json.load(f)
            self.schemas.append(schema)

        # Load the memmap file
        mmap = np.memmap(mmap_filename, dtype=dtype, mode="r")
        self.memmaps.append(mmap)

        # Update the mapping from data_id to chunk
        for entry in schema:
            self.data_id_to_chunk[entry["data_id"]] = chunk_index

    def __getitem__(self, index):
        data_id = list(self.data_id_to_chunk.keys())[index]
        chunk_idx = self.data_id_to_chunk[data_id]

        nested_dict = {}

        mmap = self.memmaps[chunk_idx]
        schema = self.schemas[chunk_idx]
        for entry in schema:
            if entry["data_id"] == data_id:
                # Read the data and put it into the nested dictionary
                path = entry["path"]
                offset = entry["offset"]
                shape = tuple(entry["shape"])
                dtype = np.dtype(entry["dtype"])

                array = np.ndarray(shape, dtype, buffer=mmap, offset=offset, order="C")
                tensor = torch.as_tensor(array)

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


def collate_nested_dicts(batch):
    # `batch` is a list of tuples, each tuple is (data_id, nested_dict)
    batched_data_ids = [data_id for data_id, _ in batch]

    # Initialize the batched_nested_dicts by deep copying the first nested_dict structure
    # Replace all tensors with lists to hold tensors from all items in the batch
    first_nested_dict = batch[0][1]
    batched_nested_dicts = {
        k: _init_collate_structure(v) for k, v in first_nested_dict.items()
    }

    # Now iterate through all items and populate the batched_nested_dicts
    for _, nested_dict in batch:
        _merge_dicts(batched_nested_dicts, nested_dict)

    # Finally, we should collate the lists of tensors into batched tensors
    _collate_tensors_in_structure(batched_nested_dicts)

    return batched_data_ids, batched_nested_dicts


def _init_collate_structure(nested_dict):
    # Initialize the collate structure based on the first item
    if isinstance(nested_dict, dict):
        return {k: _init_collate_structure(v) for k, v in nested_dict.items()}
    else:
        return []


def _merge_dicts(accumulator, new_data):
    # Merge new_data into the accumulator recursively
    for key, value in new_data.items():
        if isinstance(value, dict):
            # Recursive call if the value is a dictionary
            _merge_dicts(accumulator[key], value)
        else:
            # Assume the value is a tensor, append it to the list in accumulator
            accumulator[key].append(value)


def _collate_tensors_in_structure(nested_dict):
    # Collate all lists of tensors within the nested structure
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            # Recursive call if the value is a dictionary
            _collate_tensors_in_structure(value)
        else:
            # Stack all tensors in the list along a new batch dimension
            nested_dict[key] = default_collate(value)


ds = DefaultLogDataset("./analog")
dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_nested_dicts)
if True:
    import time

    start = time.time()
    for data_ids, data in dl:
        for key, value in data.items():
            print(f"[{key}]: {value.shape}")
    end = time.time()
    print(end - start)
