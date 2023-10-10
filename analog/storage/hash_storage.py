from analog.storage import StorageHandlerBase
from analog.storage.utils import get_world_size


def hash_tensor(tensor):
    # pytorch tensor hash is not based on the content of the tensor: https://github.com/pytorch/pytorch/issues/2569
    # use numpy as a workaround: https://stackoverflow.com/a/16592241
    numpy_tensor = tensor.detach().cpu().numpy()
    numpy_tensor.flags.writeable = False
    return hashlib.sha256(numpy_tensor.tobytes())


class HashStorageHandler(StorageHandlerBase):
    def __init__(self, config):
        super().__init__(config)

    def set_data_id(self, data_id):
        self.data_id = []
        for d in data_id:
            data_hash = hash_tensor(d)
            assert data_hash not in self.buffer and data_hash not in self.data_id
            self.data_id.append(data_hash)
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