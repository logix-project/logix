import pymongo

from analog.storage import StorageHandlerBase
from analog.utils import to_numpy
from analog.storage.utils import msgpack_serialize


class MongoDBStorageHandler(StorageHandlerBase):
    def initialize(self):
        """
        Initializes a connection to the MongoDB client and sets up the required collections.
        """
        self.client = pymongo.MongoClient(
            self.config.get("mongo_url", "mongodb://localhost:27017/")
        )
        self.db = self.client[self.config.get("db_name", "neural_logs")]
        self.logs_collection = self.db[self.config.get("collection_name", "logs")]

        self.buffer = []

    def format_log(self, module_name, log_type, data):
        """
        Formats the data in the structure needed for MongoDB.

        Args:
            module_name (str): The name of the module.
            log_type (str): The type of activation (e.g., "forward" or "backward").
            data: The data to be logged.

        Returns:
            dict: The formatted log data.
        """
        assert len(data) == len(self.data_id)

        log = []
        for datum, data_id in zip(data, self.data_id):
            log.append(
                {
                    "data_id": data_id,
                    "module_name": module_name,
                    "log_type": log_type,
                    "data": self.serialize_tensor(datum),
                }
            )
        return log

    def add(self, module_name, log_type, data):
        """
        Adds activation data to the buffer.

        Args:
            module_name (str): The name of the module.
            log_type (str): Type of log (e.g., "forward", "backward", or "grad").
            data: Data to be logged.
        """
        log = self.fotmat_log(module_name, log_type, data)
        self.buffer.extend(log)

    def push(self):
        """
        In the context of MongoDB, there's no batch operation needed like in in-memory operations.
        Data is immediately committed with insert operations.
        Thus, this method can be a placeholder or handle any finalization you need.
        """
        if self.max_buffer_size > 0 and len(self.buffer) > self.max_buffer_size:
            self.logs_collection.insert_many(self.buffer)
            self.buffer.clear()

    def serialize_tensor(self, tensor):
        """
        Serializes the given tensor.

        Args:
            tensor: The tensor to be serialized.

        Returns:
            The serialized tensor.
        """
        numpy_tensor = to_numpy(tensor)
        return msgpack_serialize(numpy_tensor)

    def finalize(self):
        self.logs_collection.insert_many(self.buffer)
