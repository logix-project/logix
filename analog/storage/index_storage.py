from analog.storage import StorageHandlerBase


class IndexStorageHandler(StorageHandlerBase):
    def __init__(self, config):
        super().__init__(config)
        self.data = {}

    def set_data_id(self, data_id):
        self.counter += len(data_id)

    def add(self, module_id, log_type, data):
        if module_id not in self.data:
            self.data[module_id] = {}
        if log_type not in self.data[module_id]:
            self.data[module_id][log_type] = []
        self.data[module_id][log_type].append(data)

    def push(self):
        pass

    def update_covariance(self, module_id, log_type, covariance):
        pass

    def disk_offload(self):
        pass

    def synchronize(self):
        pass