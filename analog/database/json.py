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
        assert len(data) == len(self.data_id)

        log = []
        for datum, data_id in zip(data, self.data_id):
            out.append(
                {
                    "data_id": data_id,
                    "module_name": module_name,
                    "activation_type": activation_type,
                    "data": serialize_tensor(datum),
                }
            )
        return log

    def push(self):
        """
        For the JSON handler, there's no batch operation needed since each add operation writes to the file.
        This can be a placeholder or used for any finalization operations.
        """
        pass

    def serialize_tensor(self, tensor):
        """
        Serializes the given tensor.

        Args:
            tensor: The tensor to be serialized.

        Returns:
            The serialized tensor.
        """
        return tensor.tolist()
