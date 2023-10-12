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

    def format_log(self, module_name, log_type, data):
        """
        Formats the data in the structure needed for Redis.

        Args:
            module_name (str): The name of the module.
            log_type (str): The type of activation (e.g., "forward" or "backward").
            data: The data to be logged.

        Returns:
            str: The formatted log data.
        """
        assert len(data) == len(self.data_id)

        log = []
        for datum, data_id in zip(data, self.data_id):
            out.append(
                {
                    "data_id": data_id,
                    "module_name": module_name,
                    "activation_type": activation_type,
                    "data": datum,
                }
            )
        return log

    def push(self):
        """
        In the context of Redis, there's no batch operation needed like in in-memory operations.
        Data is immediately committed with insert operations.
        Thus, this method can be a placeholder or handle any finalization you need.
        """
        pass
