import os
from typing import Any, List

import numpy as np
import torch
from torch.utils.data import Dataset

from analog.utils import nested_dict, to_numpy
from analog.storage import StorageHandlerBase
from analog.storage.utils import msgpack_deserialize, msgpack_serialize


class LMDBStorageHandler(StorageHandlerBase):
    def parse_config(self) -> None:
        """
        Parse the configuration parameters.
        """
        self.max_buffer_size = self.config.get("max_buffer_size", -1)
        self.file_path = self.config.get("file_path")

    def initialize(self) -> None:
        """
        Sets up the file path and prepares the JSON handler.
        Checks if the file exists, and if not, creates an initial empty JSON structure.
        """
        self.buffer = nested_dict()
        self.push_count = 0

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
            self.buffer[data_id][module_name][log_type] = to_numpy(datum)

    def push(self) -> None:
        """
        For the DefaultHandler, there's no batch operation needed since each add operation writes to the file.
        This can be a placeholder or used for any finalization operations.
        """
        if self.max_buffer_size > 0 and len(self.buffer) > self.max_buffer_size:
            save_path = str(os.path.join(self.file_path, f"data_{self.push_count}.pt"))
            torch.save(self.buffer, save_path)

            self.push_count += 1
            self.buffer.clear()

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


class LmdbLogDataset(data.Dataset):
    def __init__(self, db_path):
        self.env = lmdb.open(
            db_path, readonly=True, lock=False, readahead=False, meminit=False
        )
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            raw_data = txn.get(str(index).encode())
        data = msgpack_deserialize(raw_data)
        for idx2 in data:
            for idx3 in data[idx2]:
                data[idx2][idx3] = torch.tensor(data[idx2][idx3])
        return data
