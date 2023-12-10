import os
from typing import Any, List, Dict

import torch
from torch.utils.data import DataLoader

from analog.state import AnaLogState
from analog.storage.log_loader import DefaultLogDataset
from analog.storage.log_loader_util import collate_nested_dicts
from analog.storage.buffer_handler import BufferHandler
from analog.utils import get_logger


class StorageHandler:
    def __init__(self,
                 buffer_handler: BufferHandler = None,
                 config: Dict = None,
                 state: AnaLogState = None,
                 ):
        self.config = config

        # Default log saving config.
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')

        # Init buffer.
        if buffer_handler is None:
            self.buffer_handler = BufferHandler()
        else:
            self.buffer_handler = buffer_handler

        self.buffer_handler.set_file_prefix("log_chunk_")

        # Parse config.
        self.parse_config()

        # Precondition.
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        else:
            get_logger().warning(f"Log directory {self.log_dir} already exists.\n")

    def parse_config(self) -> None:
        """
        Parse the configuration parameters.
        """
        self.log_dir = self.config.get("log_dir", self.log_dir)
        self.buffer_handler.set_log_dir(self.log_dir)

        flush_threshold = self.config.get("flush_threshold", -1)  # -1 flushes once at the end.
        self.buffer_handler.set_flush_threshold(flush_threshold)

        max_workers = self.config.get("worker", 1)
        self.buffer_handler.set_max_worker(max_workers)

    def clear(self):
        """
        Clears the buffer.
        """
        self.buffer_handler.buffer_clear()

    def set_data_id(self, data_id):
        """
        Set the data ID for logging.

        Args:
            data_id: The ID associated with the data.
        """
        self.buffer_handler.set_data_id(data_id)

    def get_buffer(self):
        """
        Returns the buffer.

        Returns:
            dict: The buffer.
        """
        return self.buffer_handler.get_buffer()

    # TODO: Clean up before release.
    def format_log(self, module_name: str, log_type: str, data):
        """
        Formats the data in the structure needed for the JSON file.

        Args:
            module_name (str): The name of the module.
            log_type (str): The type of activation (e.g., "forward", "backward", or "grad").
            aa: The data to be logged.

        Returns:
            dict: The formatted log data.
        """
        pass

    def buffer_append(self, module_name: str, log_type: str, data) -> None:
        """
        Adds activation data to the buffer.

        Args:
            module_name (str): The name of the module.
            log_type (str): Type of log (e.g., "forward", "backward", or "grad").
            data: Data to be logged.
        """
        self.buffer_handler.buffer_append(module_name, log_type, data)

    # TODO: fix.
    def add_on_exit(self):
        """
        Add log state on exit.
        """
        log_state = self._state.log_state

        def _add(log, buffer, idx):
            for key, value in log.items():
                if isinstance(value, torch.Tensor):
                    # print(value.shape)
                    numpy_value = to_numpy(value[idx])
                    buffer[key] = numpy_value
                    self.buffer_size += numpy_value.size
                else:
                    _add(value, buffer[key], idx)

        for idx, data_id in enumerate(self.data_id):
            _add(log_state, self.buffer[data_id], idx)

    def _flush_unsafe(self, buffer, push_count) -> str:
        """
        _flush_unsafe is thread unsafe flush of current buffer. No shared variable must be allowed.
        """
        save_path = self.file_prefix + f"{push_count}.mmap"
        torch.save(buffer, save_path)
        buffer_list = [(k, v) for k, v in buffer]
        self.mmap_handler.write(buffer_list, save_path)
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
        if len(self.buffer) == 0:
            return self.log_dir
        buffer_list = [(k, v) for k, v in self.buffer.items()]
        self.mmap_handler.write(
            buffer_list, self.file_prefix + f"{self.push_count}.mmap"
        )

        self.push_count += 1
        del buffer_list
        self.buffer.clear()
        self.buffer_size = 0
        return self.log_dir

    def flush(self) -> None:
        """
        For the DefaultHandler, there's no batch operation needed since each add operation writes to the file.
        This can be a placeholder or used for any finalization operations.
        """
        self.buffer_handler.flush()

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
        self.buffer_handler.finalize()

    def _build_log_dataset(self):
        """
        Returns log dataset class.
        """

        return DefaultLogDataset(self.log_dir)

    def build_log_dataloader(self, batch_size=16, num_workers=0):
        log_dataloader = DataLoader(
            self._build_log_dataset(),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_nested_dicts,
        )
        return log_dataloader
