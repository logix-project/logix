import os
from typing import Any, List, Dict

import torch
from torch.utils.data import DataLoader

from analog.state import AnaLogState
from analog.storage.log_loader import DefaultLogDataset
from analog.storage.log_loader_util import collate_nested_dicts
from analog.storage.buffer_handler import BufferHandler
from analog.utils import get_logger, get_rank, get_world_size


class StorageHandler:
    def __init__(
        self,
        buffer_handler: BufferHandler = None,
        config: Dict = None,
        state: AnaLogState = None,
    ):
        self.log_dir = ""

        self.config = config
        self.state = state

        # Init buffer.
        if buffer_handler is None:
            self.buffer_handler = BufferHandler()
        else:
            self.buffer_handler = buffer_handler

        self.buffer_handler.set_file_prefix("log_chunk_")
        if get_world_size() > 1:
            self.buffer_handler.set_file_prefix(f"log_rank_{get_rank()}_chunk_")

        # Parse config.
        self.parse_config()

        # Precondition.
        if os.path.exists(self.log_dir):
            get_logger().warning(f"Log directory {self.log_dir} already exists.\n")

    def parse_config(self) -> None:
        """
        Parse the configuration parameters.
        """
        self.log_dir = self.config.get("log_dir")
        self.buffer_handler.set_log_dir(self.log_dir)

        flush_threshold = self.config.get(
            "flush_threshold", -1
        )  # -1 flushes once at the end.
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

    def buffer_append_on_exit(self):
        """
        Add log state on exit.
        """
        self.buffer_handler.buffer_append_on_exit(self.state.log_state)

    def flush(self) -> None:
        """
        For the DefaultHandler, there's no batch operation needed since each add operation writes to the file.
        This can be a placeholder or used for any finalization operations.
        """
        self.buffer_handler.flush()

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
