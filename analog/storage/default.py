import os
from typing import Dict

from torch.utils.data import DataLoader

from analog.batch_info import BatchInfo
from analog.storage.log_loader import DefaultLogDataset
from analog.storage.log_loader_util import collate_nested_dicts
from analog.storage.log_saver import LogSaver
from analog.utils import get_logger, get_rank, get_world_size


class StorageHandler:
    def __init__(
        self,
        log_saver: LogSaver = None,
        config: Dict = None,
        binfo: BatchInfo = None,
    ):
        self.log_dir = ""

        self.config = config
        self.binfo = binfo

        # Init buffer.
        if log_saver is None:
            self.log_saver = LogSaver()
        else:
            self.log_saver = log_saver

        self.log_saver.set_file_prefix(f"log_rank_{get_rank()}_chunk_")

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
        self.log_saver.set_log_dir(self.log_dir)

        flush_threshold = self.config.get(
            "flush_threshold", -1
        )  # -1 flushes once at the end.
        self.log_saver.set_flush_threshold(flush_threshold)

        max_workers = self.config.get("worker", 1)
        self.log_saver.set_max_worker(max_workers)

    def clear(self):
        """
        Clears the buffer.
        """
        self.log_saver.buffer_clear()

    def set_data_id(self):
        """
        Set the data ID for logging.

        Args:
            data_id: The ID associated with the data.
        """
        self.log_saver.set_data_id(self.binfo.data_id)

    def get_buffer(self):
        """
        Returns the buffer.

        Returns:
            dict: The buffer.
        """
        return self.log_saver.get_buffer()

    def buffer_write_on_exit(self):
        """
        Add log state on exit.
        """
        self.log_saver.buffer_write_on_exit(self.binfo.log)

    def flush(self) -> None:
        """
        For the DefaultHandler, there's no batch operation needed since each add operation writes to the file.
        This can be a placeholder or used for any finalization operations.
        """
        self.log_saver.flush()

    def finalize(self) -> None:
        """
        Dump everything in the buffer to a disk.
        """
        self.log_saver.finalize()

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
