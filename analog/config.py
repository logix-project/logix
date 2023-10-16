from typing import Dict, Any
import yaml

from analog.logger import get_logger


class Config:
    """
    Configuration management class.
    Loads configurations from a YAML file and provides access to component-specific configurations.
    """

    # Default values for each configuration
    _DEFAULTS = {
        "global": {},
        "storage": {"type": "default", "file_path": "./ana_log"},
        "hessian": {"type": "kfac", "damping": 1e-2},
    }

    def __init__(self, config_file: str) -> None:
        """
        Initialize Config class with given configuration file.

        :param config_file: Path to the YAML configuration file.
        """
        try:
            with open(config_file, "r") as file:
                self.data: Dict[str, Any] = yaml.safe_load(file)
        except FileNotFoundError:
            get_logger().warning("Configuration file not found. Using default values.")
            self.data = {}

    def get_global_config(self) -> Dict[str, Any]:
        """
        Retrieve global configuration.

        :return: Dictionary containing global configurations.
        """
        return self.data.get("global", self._DEFAULTS["global"])

    def get_storage_config(self) -> Dict[str, Any]:
        """
        Retrieve storage configuration.

        :return: Dictionary containing storage configurations.
        """
        return self.data.get("storage", self._DEFAULTS["storage"])

    def get_hessian_config(self) -> Dict[str, Any]:
        """
        Retrieve Hessian configuration.

        :return: Dictionary containing Hessian configurations.
        """
        return self.data.get("hessian", self._DEFAULTS["hessian"])
