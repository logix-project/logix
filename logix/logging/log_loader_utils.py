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

import logging
import os
import re
from collections import OrderedDict
from functools import reduce
from typing import List

import numpy as np
import torch
from torch.utils.data import default_collate

from logix.logging.mmap import MemoryMapHandler


def extract_rank_and_chunk(filename):
    """
    Extracts the rank and chunk index from the filename.

    Args:
        filename (str): Filename to extract rank and chunk index from.

    Returns:
        Tuple[int, int]: Tuple containing the rank and chunk index.
    """
    match = re.search(r"rank_(\d+)_chunk_(\d+)", filename)
    return int(match.group(1)), int(match.group(2))


def find_chunk_indices(path) -> List:
    """
    Finds and returns the sorted list of chunk indices based on the filenames in the input path.

    Args:
        path (str): The path to search for chunk files.

    Returns:
        List[int]: Sorted list of chunk indices.
    """
    chunk_indices = []
    for filename in os.listdir(path):
        if filename.endswith(".mmap") and filename.startswith("log_"):
            chunk_index = filename.rstrip(".mmap").strip("log_")
            chunk_indices.append(chunk_index)

    return sorted(chunk_indices, key=extract_rank_and_chunk)


def get_mmap_data(path, mmap_filename, dtype="uint8") -> List:
    """
    Adds memory-mapped files for the given mmap file.

    Args:
        path (str): Path to the directory containing the mmap file.
        mmap_filename (str): Filename of the mmap file.

    Returns:
       List: A list of memory maps.
    """
    with MemoryMapHandler.read(path, mmap_filename, dtype) as mm:
        return mm


def get_mmap_metadata(
    data_id_to_chunk, path, metadata_filename, chunk_index
) -> OrderedDict:
    metadata = MemoryMapHandler.read_metafile(path, metadata_filename)
    # Update the mapping from data_id to chunk
    for entry in metadata:
        if entry["data_id"] in data_id_to_chunk:
            logging.warning(f"duplicated data_id detected: {entry['data_id']}")
            continue
        data_id_to_chunk[entry["data_id"]] = (chunk_index, entry)
    return data_id_to_chunk


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


def get_entry_metadata(entries):
    blocks = []
    dtype = None
    for entry in entries:
        blocks.append(reduce(lambda x, y: x * y, entry["shape"]))
        dtype = np.dtype(entry["dtype"])
    blocksize = reduce(lambda x, y: x + y, blocks)
    return blocksize, dtype


def get_flatten_item(mmap, offset, block_size, dtype="float32"):
    array = np.ndarray(
        block_size,
        dtype,
        buffer=mmap,
        offset=offset,
        order="C",
    )
    return torch.from_numpy(array).clone()


def unflatten_tensor(flat_tensor, shape, start):
    num_elements = reduce(lambda x, y: x * y, shape)
    end = start + num_elements
    unflattened_tensor = flat_tensor[start:end].view(*shape)
    return unflattened_tensor, end


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
