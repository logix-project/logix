import os
from concurrent.futures import ThreadPoolExecutor
import torch

from analog.utils import get_logger, nested_dict, to_numpy
from analog.constants import GRAD
from analog.storage.utils import MemoryMapHandler


class BufferHandler:
    def __init__(self):
        self.max_worker = 0
        self.allow_async = False

        self.buffer = nested_dict()
        self.buffer_size = 0

        self.flush_count = 0
        self.flush_threshold = 0

        self.data_id = None
        self.file_prefix = ""

    def buffer_append(self, module_name: str, log_type: str, data) -> None:
        """
        Adds activation data to the buffer.

        Args:
            module_name (str): The name of the module.
            log_type (str): Type of log (e.g., "forward", "backward", or "grad").
            data: Data to be logged.
        """
        assert len(data) == len(self.data_id)
        for datum, data_id in zip(data, self.data_id):
            numpy_datum = to_numpy(datum)
            if log_type == GRAD:
                if module_name not in self.buffer[data_id]:
                    self.buffer[data_id][module_name] = numpy_datum
                else:
                    self.buffer[data_id][module_name] += numpy_datum
            else:
                if log_type not in self.buffer[data_id][module_name]:
                    self.buffer[data_id][module_name][log_type] = numpy_datum
                else:
                    self.buffer[data_id][module_name][log_type] += numpy_datum
            self.buffer_size += numpy_datum.size

    def _flush_unsafe(self, log_dir, buffer, flush_count) -> str:
        """
        _flush_unsafe is thread unsafe flush of current buffer. No shared variable must be allowed.
        """
        filename = self.file_prefix + f"{flush_count}.mmap"
        buffer_list = [(k, v) for k, v in buffer]
        MemoryMapHandler.write(log_dir, filename, buffer_list)
        return filename

    def _flush_safe(self, log_dir) -> str:
        """
        _flush_safe is thread safe flush of current buffer.
        """
        buffer_copy = self.buffer.copy()
        flush_count_copy = self.flush_count
        self.flush_count += 1
        self.buffer.clear()
        self.buffer_size = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            save_path = executor.submit(
                self._flush_unsafe, log_dir, buffer_copy, flush_count_copy
            )
        return save_path

    def _flush_serialized(self, log_dir) -> str:
        """
        _flush_serialized executes the flushing of the buffers in serialized manner.
        """
        if len(self.buffer) == 0:
            return log_dir
        buffer_list = [(k, v) for k, v in self.buffer.items()]
        MemoryMapHandler.write(
            log_dir, self.file_prefix + f"{self.flush_count}.mmap", buffer_list, dtype="uint8"
        )

        self.flush_count += 1
        self.buffer_clear()
        return log_dir

    def flush(self, log_dir) -> None:
        """
        For the DefaultHandler, there's no batch operation needed since each add operation writes to the file.
        This can be a placeholder or used for any finalization operations.
        """
        if 0 < self.flush_threshold < self.buffer_size:
            if self.allow_async:
                self._flush_safe(log_dir)
                return
            self._flush_serialized(log_dir)

    def finalize(self, log_dir):
        if self.buffer_size > 0:
            self._flush_serialized(log_dir)

    def set_data_id(self, data_id):
        self.data_id = data_id

    def set_max_worker(self, max_worker):
        self.max_worker = max_worker
        self.allow_async = True if self.max_worker > 1 else False

    def set_file_prefix(self, file_prefix):
        self.file_prefix = file_prefix

    def set_flush_threshold(self, flush_threshold):
        self.flush_threshold = flush_threshold

    def get_buffer(self):
        return self.buffer

    def buffer_clear(self):
        self.buffer.clear()
        self.buffer_size = 0
