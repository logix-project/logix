from abc import ABC, abstractmethod


class StorageHandlerBase(ABC):
    def __init__(self, config=None):
        """
        Initializes the StorageHandlerBase.

        Args:
            config (dict, optional): Configuration parameters for the handler.
        """
        self.config = config
        self.buffer = None

        self.parse_config()
        self.initialize()

    @abstractmethod
    def parse_config(self):
        """
        Abstract method to parse the configuration.
        """
        pass

    @abstractmethod
    def initialize(self):
        """
        Abstract method to handle initial setup or connections.
        """
        pass

    @abstractmethod
    def finalize(self):
        """
        Abstract method to finalize the StorageHandler.
        """
        pass

    @abstractmethod
    def format_log(self, module_name: str, log_type: str, data):
        """
        Abstract method to format the logging data.

        Args:
            module_name (str): The name of the module.
            log_type (str): The type of activation (e.g., "forward", "backward", or "grad").
            data: The data to be logged.

        Returns:
            The formatted log data.
        """
        pass

    @abstractmethod
    def add(self, module_name: str, log_type: str, data, mask=None):
        """
        Adds activation data to the buffer.

        Args:
            module_name (str): The name of the module.
            log_type (str): Type of log (e.g., "forward", "backward", or "grad").
            data: Data to be logged.
        """
        pass

    @abstractmethod
    def push(self):
        """
        Abstract method to push the stored data to the destination (e.g., database or file).
        """
        pass

    @abstractmethod
    def query(self, data_id):
        """
        Abstract method to query the destination for data.
        """
        pass

    @abstractmethod
    def query_batch(self, data_ids):
        """
        Abstract method to query the destination for data.
        """
        pass

    @abstractmethod
    def build_log_dataloader(self):
        """
        Abstract method to build the log dataloader.
        """
        pass

    @abstractmethod
    def serialize_tensor(self, tensor):
        """
        Abstract method to serialize a tensor.

        Args:
            tensor: The tensor to be serialized.

        Returns:
            The serialized tensor.
        """
        pass

    def clear(self):
        """
        Clears the buffer.
        """
        self.buffer.clear()

    def set_data_id(self, data_id):
        """
        Set the data ID for logging.

        Args:
            data_id: The ID associated with the data.
        """
        self.data_id = data_id

    def get_buffer(self):
        """
        Returns the buffer.

        Returns:
            dict: The buffer.
        """
        return self.buffer
