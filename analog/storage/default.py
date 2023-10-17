import numpy as np

from analog.utils import nested_dict
from analog.storage import StorageHandlerBase


class DefaultStorageHandler(StorageHandlerBase):
    def initialize(self):
        """
        Sets up the file path and prepares the JSON handler.
        Checks if the file exists, and if not, creates an initial empty JSON structure.
        """
        self.buffer = nested_dict()
        self.push_count = 0

        # config
        # TODO: create parse_config method
        self.max_buffer_size = self.config.get("max_buffer_size", -1)
        self.file_path = self.config.get("file_path")

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
            self.buffer[data_id][module_name][log_type] = datum.cpu()

    def push(self):
        """
        For the JSON handler, there's no batch operation needed since each add operation writes to the file.
        This can be a placeholder or used for any finalization operations.
        """
        if self.max_buffer_size > 0 and len(self.buffer) > self.max_buffer_size:
            np.savez(f"{self.file_path}/data_{self.push_count}.npz", **self.buffer)

            self.push_count += 1
            self.buffer = nested_dict()

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
        np.savez(f"{self.file_path}/data_{self.push_count}.npz", **self.buffer)

