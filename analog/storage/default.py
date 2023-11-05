import os
from typing import Any, List
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from collections import OrderedDict

import json
import numpy as np
import torch
from torch.utils.data import Dataset

from analog.utils import nested_dict, to_numpy
from analog.storage import StorageHandlerBase
from analog.utils import stack_tensor
from analog.constants import GRAD


class DefaultStorageHandler(StorageHandlerBase):
    def parse_config(self) -> None:
        """
        Parse the configuration parameters.
        """
        self.flush_threshold = self.config.get("flush_threshold", -1)
        self.file_path = self.config.get("file_path", "analog")
        self.max_workers = self.config.get("worker", 0)
        self.allow_async = True if self.max_workers > 1 else False

    def initialize(self) -> None:
        """
        Sets up the file path and prepares the JSON handler.
        Checks if the file exists, and if not, creates an initial empty JSON structure.
        """
        self.buffer = nested_dict()
        self.push_count = 0
        if self.allow_async:
            self.lock = Lock()

    def format_log(self, module_name: str, log_type: str, data):
        """
        Formats the data in the structure needed for the JSON file.

        Args:
            module_name (str): The name of the module.
            log_type (str): The type of activation (e.g., "forward", "backward", or "grad").
            data: The data to be logged.

        Returns:
            dict: The formatted log data.
        """
        pass

    def add(self, module_name: str, log_type: str, data) -> None:
        """
        Adds activation data to the buffer.

        Args:
            module_name (str): The name of the module.
            log_type (str): Type of log (e.g., "forward", "backward", or "grad").
            data: Data to be logged.
        """
        assert len(data) == len(self.data_id)
        for datum, data_id in zip(data, self.data_id):
            if log_type == GRAD:
                self.buffer[data_id][module_name] = to_numpy(datum)
            else:
                self.buffer[data_id][module_name][log_type] = to_numpy(datum)
            self.buffer_size += data.size

    def _flush_unsafe(self, buffer, push_count) -> str:
        """
        _flush_unsafe is thread unsafe flush of current buffer. No shared variable must be allowed.
        """
        save_path = str(os.path.join(self.file_path, f"data_{push_count}.pt"))
        torch.save(buffer, save_path)
        return save_path

    def _flush_safe(self) -> str:
        """
        _flush_safe is thread safe flush of current buffer.
        """
        buffer_copy = self.buffer.copy()
        push_count_copy = self.push_count
        self.push_count += 1
        self.buffer.clear()
        self.buffer_size = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            save_path = executor.submit(
                self._flush_unsafe, buffer_copy, push_count_copy
            )
        return save_path

    def _flush_serialized(self) -> str:
        """
        _flush_serialized executes the flushing of the buffers in serialized manner.
        """
        save_path = str(os.path.join(self.file_path, f"data_{self.push_count}.pt"))
        torch.save(self.buffer, save_path)
        self.push_count += 1
        self.buffer.clear()
        self.buffer_size = 0
        return save_path

    def flush(self) -> None:
        """
        For the DefaultHandler, there's no batch operation needed since each add operation writes to the file.
        This can be a placeholder or used for any finalization operations.
        """
        if 0 < self.flush_threshold < self.buffer_size:
            if self.allow_async:
                self._flush_safe()
                return
            self._flush_serialized()

    def verify_flush(self):
        """
        verify flush somehow..
        """
        raise NotImplementedError

    def query(self, data_id: Any):
        """
        Query the data with the given data ID.

        Args:
            data_id: The data ID.

        Returns:
            The queried data.
        """
        return self.buffer[data_id]

    def query_batch(self, data_ids: List[Any]):
        """
        Query the data with the given data IDs.

        Args:
            data_ids: The data IDs.

        Returns:
            The queried data.
        """
        return [self.buffer[data_id] for data_id in data_ids]

    def serialize_tensor(self, tensor: torch.Tensor):
        """
        Serializes the given tensor.

        Args:
            tensor: The tensor to be serialized.

        Returns:
            The serialized tensor.
        """
        pass

    def finalize(self) -> None:
        """
        Dump everything in the buffer to a disk.
        """
        save_path = str(os.path.join(self.file_path, f"data_{self.push_count}.pt"))
        torch.save(self.buffer, save_path)

    def build_log_dataloader(self):
        pass


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
            schema_filename = os.path.join(log_dir, f"metadata_chunk_{chunk_index}.json") # does the different chunks have different schemas?
            self._add_schema_and_mmap(schema_filename, mmap_filename, chunk_index)

    def _find_chunk_indices(self, directory):
        chunk_indices = []
        for filename in os.listdir(directory):
            if filename.endswith(".mmap"):
                parts = filename.rstrip('.mmap').split('_')
                if len(parts) != 0:
                    chunk_index = parts[-1]
                chunk_indices.add(int(chunk_index))
        return sorted(chunk_indices)

    def _add_schema_and_mmap(self, schema_filename, mmap_filename, chunk_index, dtype='uint8'):
        # Load the schema
        with open(schema_filename, 'r') as f:
            schema = json.load(f)
            self.schemas.append(schema)
        
        # Load the memmap file
        mmap = np.memmap(mmap_filename, dtype=dtype, mode='r')
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

                array = np.ndarray(
                    shape, dtype, buffer=mmap, offset=offset, order="C"
                )
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
