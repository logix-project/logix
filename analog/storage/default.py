import os
from typing import Any, List
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

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
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            save_path = executor.submit(self._flush_unsafe, buffer_copy, push_count_copy)
        return save_path

    def _flush_serialized(self) -> str:
        """
        _flush_serialized executes the flushing of the buffers in serialized manner.
        """
        save_path = str(os.path.join(self.file_path, f"data_{self.push_count}.pt"))
        torch.save(self.buffer, save_path)
        self.push_count += 1
        self.buffer.clear()
        return save_path

    def flush(self) -> None:
        """
        For the DefaultHandler, there's no batch operation needed since each add operation writes to the file.
        This can be a placeholder or used for any finalization operations.
        """
        if 0 < self.flush_threshold < len(self.buffer):
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
