import json
import os


class JSONStorageHandler(AbstractStorageHandler):
    def initialize(self):
        """
        Sets up the file path and prepares the JSON handler.
        Checks if the file exists, and if not, creates an initial empty JSON structure.
        """
        self.file_path = self.config.get("json_file_path", "logs.json")
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as file:
                json.dump({}, file)

    def set_data_id(self, data_id):
        """
        Set the data ID for subsequent logging.

        Args:
            data_id: The ID associated with the data.
        """
        self.data_id = data_id

    def format_log(self, module_name, activation_type, data):
        """
        Formats the data in the structure needed for the JSON file.

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
        Adds activation data to the JSON file.

        Args:
            module_name (str): The name of the module.
            activation_type (str): Type of activation (e.g., "forward", "backward").
            activation: Activation data to be logged.
        """
        log_data = self.format_log(module_name, activation_type, activation)
        with open(self.file_path, "r+") as file:
            data = json.load(file)
            data[self.data_id] = log_data
            file.seek(0)
            json.dump(data, file, indent=4)

    def add_covariance(self, module_name, activation_type, covariance):
        """
        Adds covariance data to the JSON file.

        Args:
            module_name (str): The name of the module.
            activation_type (str): The type of activation (e.g., "forward" or "backward").
            covariance: The covariance data.
        """
        # This example assumes covariance is saved similarly to activations.
        formatted_data = self.format_log(module_name, activation_type, covariance)

    def clear(self):
        """
        Clear the stored data in the JSON file.
        """
        with open(self.file_path, "w") as file:
            json.dump({}, file)

    def push(self):
        """
        For the JSON handler, there's no batch operation needed since each add operation writes to the file.
        This can be a placeholder or used for any finalization operations.
        """
        pass
