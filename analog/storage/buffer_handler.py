from concurrent.futures import ThreadPoolExecutor
import torch

from analog.utils import nested_dict, to_numpy
from analog.storage.utils import MemoryMapHandler


class BufferHandler:
    def __init__(self):
        self.log_dir = ""
        self.max_worker = 0
        self.allow_async = False

        self.buffer = nested_dict()
        self.buffer_size = 0

        self.flush_count = 0
        self.flush_threshold = 0

        self.data_id = None
        self.file_prefix = ""

    def buffer_append_on_exit(self, log_state):
        """
        Add log state on exit.
        """

        def _add(log, buffer, idx):
            for key, value in log.items():
                if isinstance(value, torch.Tensor):
                    numpy_value = to_numpy(value[idx])
                    buffer[key] = numpy_value
                    self.buffer_size += numpy_value.size
                    continue
                _add(value, buffer[key], idx)

        for idx, data_id in enumerate(self.data_id):
            _add(log_state, self.buffer[data_id], idx)

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
            log_dir,
            self.file_prefix + f"{self.flush_count}.mmap",
            buffer_list,
            dtype="uint8",
        )

        self.flush_count += 1
        self.buffer_clear()
        del buffer_list
        return log_dir

    def flush(self) -> None:
        """
        For the DefaultHandler, there's no batch operation needed since each add operation writes to the file.
        This can be a placeholder or used for any finalization operations.
        """
        if 0 < self.flush_threshold < self.buffer_size:
            if self.allow_async:
                self._flush_safe(self.log_dir)
                return
            self._flush_serialized(self.log_dir)

    def finalize(self):
        self._flush_serialized(self.log_dir)

    def set_data_id(self, data_id):
        self.data_id = data_id

    def set_max_worker(self, max_worker):
        self.max_worker = max_worker
        self.allow_async = True if self.max_worker > 1 else False

    def set_file_prefix(self, file_prefix):
        self.file_prefix = file_prefix

    def set_flush_threshold(self, flush_threshold):
        self.flush_threshold = flush_threshold

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir

    def get_buffer(self):
        return self.buffer

    def buffer_clear(self):
        self.buffer.clear()
        self.buffer_size = 0
