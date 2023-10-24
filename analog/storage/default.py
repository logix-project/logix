import os
from typing import Any, List

import numpy as np
import torch
from torch.utils.data import Dataset

from analog.utils import nested_dict, to_numpy
from analog.storage import StorageHandlerBase
from analog.utils import stack_tensor


class DefaultStorageHandler(StorageHandlerBase):
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

    def build_log_dataloader(self):
        pass


class DefaultLogDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.startswith("data_") and f.endswith(".pt")
            ]
        )
        self.total_samples = sum(
            [torch.load(os.path.join(data_dir, f)).shape[0] for f in self.data_files]
        )

    def __len__(self):
        return self.total_samples
