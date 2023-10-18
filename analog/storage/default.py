import os

import numpy as np
import torch

from analog.utils import nested_dict, to_numpy
from analog.storage import StorageHandlerBase


class DefaultStorageHandler(StorageHandlerBase):
    def parse_config(self):
        self.max_buffer_size = self.config.get("max_buffer_size", -1)
        self.file_path = self.config.get("file_path")

    def initialize(self):
        """
        Sets up the file path and prepares the JSON handler.
        Checks if the file exists, and if not, creates an initial empty JSON structure.
        """
        self.buffer = nested_dict()
        self.push_count = 0

    def format_log(self, module_name, log_type, data):
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

    def add(self, module_name, log_type, data):
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

    def push(self):
        """
        For the JSON handler, there's no batch operation needed since each add operation writes to the file.
        This can be a placeholder or used for any finalization operations.
        """
        if self.max_buffer_size > 0 and len(self.buffer) > self.max_buffer_size:
            save_path = str(os.path.join(self.file_path, f"data_{self.push_count}.pt"))
            torch.save(self.buffer, save_path)

            self.push_count += 1
            self.buffer.clear()

    def serialize_tensor(self, tensor):
        """
        Serializes the given tensor.

        Args:
            tensor: The tensor to be serialized.

        Returns:
            The serialized tensor.
        """
        pass

    def finalize(self):
        save_path = str(os.path.join(self.file_path, f"data_{self.push_count}.pt"))
        torch.save(self.buffer, save_path)
