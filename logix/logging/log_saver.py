# Copyright 2023-present the LogIX team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ThreadPoolExecutor

import torch

from logix.logging.mmap import MemoryMapHandler
from logix.utils import get_rank, nested_dict, to_numpy


class LogSaver:
    def __init__(self, config, state):
        self.log_dir = config.log_dir
        self.state = state
        self.model_module = self.state.get_state("model_module")
        self.file_prefix = f"log_rank_{get_rank()}_chunk_"

        self.max_worker = config.num_workers
        self.allow_async = True if self.max_worker > 1 else False

        self.flush_threshold = config.flush_threshold
        self.flush_count = 0

        self.buffer = nested_dict()
        self.buffer_size = 0

    def buffer_write(self, binfo):
        """
        Add log state on exit.
        """
        data_id = binfo.data_id
        log = binfo.log

        def _add(log, buffer, idx):
            for key, value in log.items():
                if isinstance(value, torch.Tensor):
                    numpy_value = to_numpy(value[idx])
                    buffer[key] = numpy_value
                    self.buffer_size += numpy_value.size
                    continue
                _add(value, buffer[key], idx)

        for idx, did in enumerate(data_id):
            _add(log, self.buffer[did], idx)

    def _flush_unsafe(self, log_dir, buffer, flush_count) -> str:
        """
        _flush_unsafe is thread unsafe flush of current buffer. No shared variable must be allowed.
        """
        filename = self.file_prefix + f"{flush_count}.mmap"
        buffer_list = [(k, v) for k, v in buffer]
        MemoryMapHandler.write(
            log_dir, filename, buffer_list, self.model_module["path"]
        )
        return filename

    def _flush_safe(self, log_dir) -> str:
        """
        _flush_safe is thread safe flush of current buffer.
        """
        buffer_copy = self.buffer.copy()
        flush_count_copy = self.flush_count
        self.flush_count += 1
        self.buffer_clear()
        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
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
            self.model_module["path"],
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
        """
        Dump everything in the buffer to disk when `logix.finalize()` is called.
        """
        self._flush_serialized(self.log_dir)

    def buffer_clear(self):
        """
        Clears the buffer.
        """
        self.buffer.clear()
        self.buffer_size = 0
