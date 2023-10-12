import redis
import json


class RedisStorageHandler(AbstractStorageHandler):
    def initialize(self):
        """
        Initializes a connection to the Redis client.
        """
        self.client = redis.StrictRedis(
            host=self.config.get("redis_host", "localhost"),
            port=self.config.get("redis_port", 6379),
            db=self.config.get("redis_db", 0),
        )

    def set_data_id(self, data_id):
        """
        Set the data ID for subsequent logging.

        Args:
            data_id: The ID associated with the data.
        """
        self.data_id = data_id

    def format_log(self, module_name, activation_type, data):
        """
        Formats the data in the structure needed for Redis.

        Args:
            module_name (str): The name of the module.
            activation_type (str): The type of activation (e.g., "forward" or "backward").
            data: The data to be logged.

        Returns:
            str: The formatted log data.
        """
        log = {
            "data_id": self.data_id,
            "module_name": module_name,
            "activation_type": activation_type,
            "data": data,
        }
        return json.dumps(log)

    def add_activation(self, module_name, activation_type, activation):
        """
        Adds activation data to Redis.

        Args:
            module_name (str): The name of the module.
            activation_type (str): Type of activation (e.g., "forward", "backward").
            activation: Activation data to be logged.
        """
        key = f"{module_name}:{activation_type}:{self.data_id}"
        value = self.format_log(module_name, activation_type, activation)
        self.client.set(key, value)

    def add_covariance(self, module_name, activation_type, covariance):
        """
        Adds covariance data to Redis.

        Args:
            module_name (str): The name of the module.
            activation_type (str): The type of activation (e.g., "forward" or "backward").
            covariance: The covariance data.
        """
        # This example assumes covariance is saved similarly to activations.
        formatted_data = self.format_log(module_name, activation_type, covariance)

    def clear(self):
        """
        Clear all the stored data in Redis.
        BE CAUTIOUS: This deletes everything in the chosen Redis DB.
        """
        self.client.flushdb()

    def push(self):
        """
        In the context of Redis, there's no batch operation needed like in in-memory operations.
        Data is immediately committed with insert operations.
        Thus, this method can be a placeholder or handle any finalization you need.
        """
        pass
