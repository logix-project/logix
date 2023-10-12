class DatabaseHandlerBase(ABC):
    def __init__(self, config=None):
        """
        Initializes the DatabaseHandlerBase.

        Args:
            config (dict, optional): Configuration parameters for the handler.
        """
        self.config = config
        self.initialize()

    @abstractmethod
    def initialize(self):
        """
        Abstract method to handle initial setup or connections.
        """
        pass

    @abstractmethod
    def set_data_id(self, data_id):
        """
        Abstract method to set the data ID for logging.

        Args:
            data_id: The ID associated with the data.
        """
        pass

    @abstractmethod
    def format_log(self, module_name, activation_type, data):
        """
        Abstract method to format the logging data.

        Args:
            module_name (str): The name of the module.
            activation_type (str): The type of activation (e.g., "forward" or "backward").
            data: The data to be logged.

        Returns:
            The formatted log data.
        """
        pass

    @abstractmethod
    def add_activation(self, module_name, activation_type, activation):
        """
        Abstract method to add activation data.

        Args:
            module_name (str): The name of the module.
            activation_type (str): The type of activation (e.g., "forward" or "backward").
            activation: The activation data.
        """
        pass

    @abstractmethod
    def add_covariance(self, module_name, activation_type, covariance):
        """
        Abstract method to add covariance data.

        Args:
            module_name (str): The name of the module.
            activation_type (str): The type of activation (e.g., "forward" or "backward").
            covariance: The covariance data.
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Abstract method to clear stored data.
        """
        pass

    @abstractmethod
    def push(self):
        """
        Abstract method to push the stored data to the destination (e.g., database or file).
        """
        pass
