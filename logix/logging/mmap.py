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

import json
import os
from contextlib import contextmanager
from functools import reduce

import numpy as np


def extract_arrays(obj, base_path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from extract_arrays(v, base_path + (k,))
    elif isinstance(obj, np.ndarray):
        yield base_path, obj


def get_from_nested_dict(nested_dict, keys):
    current_level = nested_dict
    for key in keys:
        if key in current_level:
            current_level = current_level[key]
    return current_level


class MemoryMapHandler:
    @staticmethod
    def write(save_path, filename, data_buffer, write_order_key, dtype="uint8"):
        file_root, file_ext = os.path.splitext(filename)
        if file_ext == "":
            filename += ".mmap"

        mmap_filename = os.path.join(save_path, filename)
        metadata_filename = os.path.join(save_path, file_root + "_metadata.json")

        total_size = sum(
            arr.nbytes for _, d in data_buffer for _, arr in extract_arrays(d)
        )
        mmap = np.memmap(mmap_filename, dtype=dtype, mode="w+", shape=(total_size,))

        metadata = []
        offset = 0
        for data_id, nested_dict in data_buffer:
            data_level_offset = offset
            data_shape = []
            block_size = 0
            # Enforcing the insert order based on the module path.
            for key in write_order_key:
                arr = get_from_nested_dict(nested_dict, key)
                bytes = arr.nbytes
                data_shape.append(arr.shape)
                block_size += reduce(lambda x, y: x * y, arr.shape)
                mmap[offset : offset + bytes] = arr.ravel().view(dtype)
                offset += arr.nbytes
            metadata.append(
                {
                    "data_id": data_id,
                    "path": write_order_key,
                    "offset": data_level_offset,
                    "shape": data_shape,
                    "block_size": block_size,
                    "dtype": str(arr.dtype),
                }
            )

        mmap.flush()
        del mmap  # Release the memmap object

        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    @contextmanager
    def read(path, filename, dtype="uint8"):
        """
        read reads the file by chunk index, it will return the data_buffer with metadata.
        Arg:
            filename (str): filename for the path to mmap.
        Returns:
            mmap (np.mmap): memory mapped buffer read from filename.
        """
        _, file_ext = os.path.splitext(filename)
        if file_ext == "":
            filename += ".mmap"

        mmap = np.memmap(os.path.join(path, filename), dtype=dtype, mode="r+")
        try:
            yield mmap
        finally:
            del mmap

    @staticmethod
    def read_metafile(path, meta_filename):
        _, file_ext = os.path.splitext(meta_filename)
        if file_ext == "":
            meta_filename += ".json"
        f = open(os.path.join(path, meta_filename), "r")
        metadata = json.load(f)  # This throws error when file does not exist.
        f.close()
        return metadata
