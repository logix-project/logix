from typing import Dict, Any
import yaml

from analog.logger import get_logger

class Config:
    """
    Configuration management class.
    Loads configurations from a YAML file and provides access to component-specific configurations.
    """

    # Default values for each configuration
    DEFAULTS = {}
    
    def __init__(self, config_file: str) -> None:
        """
        Initialize Config class with given configuration file.

        :param config_file: Path to the YAML configuration file.
        """
        try:
            with open(config_file, 'r') as file:
                self.data: Dict[str, Any] = yaml.safe_load(file)
        except FileNotFoundError:
            get_logger().warning("Configuration file not found. Using default values.")
            self.data = {}

    def get_global_config(self) -> Dict[str, Any]:
        """
        Retrieve global configuration.

        :return: Dictionary containing global configurations.
        """
        return self.data.get('global', self.DEFAULTS['global'])

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Retrieve logging configuration.

        :return: Dictionary containing logging configurations.
        """
        return self.data.get('logging', self.DEFAULTS['logging'])

    def get_storage_config(self) -> Dict[str, Any]:
        """
        Retrieve storage configuration.

        :return: Dictionary containing storage configurations.
        """
        return self.data.get('storage', self.DEFAULTS['storage'])

    def get_hessian_config(self) -> Dict[str, Any]:
        """
        Retrieve Hessian configuration.

        :return: Dictionary containing Hessian configurations.
        """
        return self.data.get('hessian', self.DEFAULTS['hessian'])
