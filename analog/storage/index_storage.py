from analog.storage import StorageHandlerBase
from analog.storage.utils import get_world_size


class IndexStorageHandler(StorageHandlerBase):
    def __init__(self, config):
        super().__init__(config)

    def set_data_id(self, data_id):
        self.data_id = [
            str(i) for i in range(self.counter, self.counter + len(data_id))
        ]
        self.counter += len(data_id)

    def add(self, module_id, log_type, data):
        assert len(data) == len(self.data_id)
        for i, data_id in enumerate(self.data_id):
            self.buffer[data_id][module_id][log_type] = data[i]

    def push(self):
        return

    def synchronize(self):
        if get_world_size() > 1:
            for module_id in self.covariance:
                for log_type in self.covariance[module_id]:
                    cov = self.covariance[module_id][log_type]
                    dist.all_reduce(cov)
                    self.covariance[module_id][log_type] = cov / get_world_size()
