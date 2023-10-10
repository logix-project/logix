import abc


class NestedDict:
    """Manage buffer on GPU and CPU. Interface like a dictionary."""

    def __init__(self):
        self.buffer = self._get_nested_defaultdict()

    @staticmethod
    def _get_nested_defaultdict():
        """Return arbitrary depth dictionary"""
        nested_dict = lambda: defaultdict(nested_dict)
        return nested_dict()

    def __getitem__(self, key):
        value = self.buffer[key]
        return value

    def __setitem__(self, key, value):
        self.buffer[key] = value

    def keys(self):
        """Return keys from gpu and cpu buffer"""
        return self.buffer.keys()

    def __contains__(self, key):
        return key in self.buffer


class StorageHandlerBase:
    def __init__(self, config):
        self.config = config

        self.data_id = None
        self.counter = 0

        self.buffer = NestedDict()
        self.covariance = NestedDict()

    @abc.abstractmethod
    def set_data_id(self, data_id):
        """Set the data id for a new batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def add(self, module_id, log_type, data):
        """Add data for the corresponding module and log type to the storage."""
        raise NotImplementedError

    @abc.abstractmethod
    def push(self):
        """Push the data to the storage."""
        raise NotImplementedError

    def clear(self):
        """Clear the buffer."""
        self.data_id = None

    def update_covariance(self, module_id, log_type, covariance, ema=False):
        """Update the covariance for the corresponding module and log type."""
        beta = (
            self.config.ema_beta
            if ema
            else self.counter / (self.counter + len(self.data_id))
        )
        self.covariance[module_id][log_type] = (
            beta * self.covariance[module_id][log_type] + (1 - beta) * covariance
        )

    def synchronize(self):
        """Synchronize logged data across devices."""
        raise NotImplementedError
