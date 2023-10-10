from abc import ABC


class StorageHandlerBase(ABC):
    def __init__(self, config):
        self.config = config

        self.counter = 0

    def set_data_id(self, data_id):
        """Set the data id for a new batch."""
        raise NotImplementedError

    def add(self, module_id, log_type, data):
        """Add data for the corresponding module and log type to the storage."""
        raise NotImplementedError

    def push(self):
        """Push the data to the storage."""
        raise NotImplementedError

    def update_covariance(self, module_id, log_type, covariance):
        """Update the covariance for the corresponding module and log type."""
        raise NotImplementedError
    
    def disk_offload(self):
        """Offload the data from CPU to the disk."""
        raise NotImplementedError

    def synchronize(self):
        """Synchronize logged data across devices."""
        raise NotImplementedError