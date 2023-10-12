import pymongo


class MongoDBStorageHandler(AbstractStorageHandler):
    def initialize(self):
        """
        Initializes a connection to the MongoDB client and sets up the required collections.
        """
        self.client = pymongo.MongoClient(
            self.config.get("mongo_url", "mongodb://localhost:27017/")
        )
        self.db = self.client[self.config.get("db_name", "neural_logs")]
        self.logs_collection = self.db[self.config.get("collection_name", "logs")]

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
        In the context of MongoDB, there's no batch operation needed like in in-memory operations.
        Data is immediately committed with insert operations.
        Thus, this method can be a placeholder or handle any finalization you need.
        """
        pass
