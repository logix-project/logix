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

    def set_data_id(self, data_id):
        """
        Set the data ID for subsequent logging.

        Args:
            data_id: The ID associated with the data.
        """
        self.data_id = data_id

    def format_log(self, module_name, activation_type, data):
        """
        Formats the data in the structure needed for MongoDB.

        Args:
            module_name (str): The name of the module.
            activation_type (str): The type of activation (e.g., "forward" or "backward").
            data: The data to be logged.

        Returns:
            dict: The formatted log data.
        """
        return {
            "data_id": self.data_id,
            "module_name": module_name,
            "activation_type": activation_type,
            "data": data,
        }

    def add_activation(self, module_name, activation_type, activation):
        """
        Adds activation data to the MongoDB.

        Args:
            module_name (str): The name of the module.
            activation_type (str): Type of activation (e.g., "forward", "backward").
            activation: Activation data to be logged.
        """
        formatted_data = self.format_log(module_name, activation_type, activation)
        self.logs_collection.insert_one(formatted_data)

    def add_covariance(self, module_name, activation_type, covariance):
        """
        Adds covariance data to the MongoDB.

        Args:
            module_name (str): The name of the module.
            activation_type (str): The type of activation (e.g., "forward" or "backward").
            covariance: The covariance data.
        """
        # This example assumes covariance is saved similarly to activations.
        formatted_data = self.format_log(module_name, activation_type, covariance)
        self.logs_collection.insert_one(formatted_data)

    def clear(self):
        """
        Clear all the stored data in the MongoDB collection.
        """
        self.logs_collection.delete_many({})

    def push(self):
        """
        In the context of MongoDB, there's no batch operation needed like in in-memory operations.
        Data is immediately committed with insert operations.
        Thus, this method can be a placeholder or handle any finalization you need.
        """
        pass
