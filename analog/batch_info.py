from analog.utils import nested_dict


class BatchInfo:
    def __init__(self, flatten):
        self.data_id = None
        self.mask = None
        self.log = nested_dict()
        self.flatten_context = None

    class FlattenContext:
        def __init__(self, paths, block_size, dtype):
            self._paths = paths
            self._block_size = block_size
            self._dtype = dtype

        @property
        def paths(self):
            return self._paths

        @property
        def block_size(self):
            return self._block_size

        @property
        def dtype(self):
            return self._dtype

    def set_flatten_context(self, paths, block_size, dtype="float32"):
        self.flatten_context = self.FlattenContext(paths, block_size, dtype)

    def clear(self):
        self.data_id = None
        self.mask = None
        self.log.clear()
