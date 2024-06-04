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

from collections import OrderedDict

from torch.utils.data import Dataset

from logix.logging.log_loader_utils import (
    find_chunk_indices,
    get_flatten_item,
    get_mmap_data,
    get_mmap_metadata,
    unflatten_tensor,
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
        flat_tensor = get_flatten_item(
            mmap, offset, entry["block_size"], entry["dtype"]
        )
        if self.flatten:
            return data_id, flat_tensor
        start = 0
        for i in range(len(entry["path"])):
            path = entry["path"][i]
            shape = tuple(entry["shape"][i])
            tensor, start = unflatten_tensor(flat_tensor, shape, start)
            current_level = nested_dict
            for key in path[:-1]:
                if key not in current_level:
                    current_level[key] = {}
                current_level = current_level[key]
            current_level[path[-1]] = tensor
        assert (
            entry["block_size"] == start
        ), f"block_size does not match with the shape for data_id: {entry['data_id']}"
        return data_id, nested_dict

    def __len__(self):
        return len(self.data_id_to_chunk)

    def close(self):
        for mmap in self.memmaps:
            del mmap
