import os
from typing import Dict, Any
import yaml

from analog.utils import get_logger


class Config:
    """
    Configuration management class.
    Loads configurations from a YAML file and provides access to component-specific configurations.
    """

    # Default values for each configuration
    _DEFAULTS = {
        "global": {"log_root": "./analog"},
        "logging": {},
        "storage": {"type": "default"},
        "hessian": {"type": "kfac", "damping": 1e-2},
        "analysis": {},
        "lora": {"init": "pca", "rank": 64},
    }

    def __init__(self, config_file: str, project_name: str) -> None:
        """
        Initialize Config class with given configuration file.

        :param config_file: Path to the YAML configuration file.
        """
        self.project_name = project_name
        try:
            with open(config_file, "r") as file:
                self.data: Dict[str, Any] = yaml.safe_load(file)
        except FileNotFoundError:
            get_logger().warning(
                "Configuration file not found. Using default values.\n"
            )
            self.data = {}

        self._global_config = self.data.get("global", self._DEFAULTS["global"])
        self._logging_config = self.data.get("logging", self._DEFAULTS["logging"])
        self._storage_config = self.data.get("storage", self._DEFAULTS["storage"])
        self._hessian_config = self.data.get("hessian", self._DEFAULTS["hessian"])
        self._analysis_config = self.data.get("analysis", self._DEFAULTS["analysis"])
        self._lora_config = self.data.get("lora", self._DEFAULTS["lora"])

        self._set_log_dir()

    def get_global_config(self) -> Dict[str, Any]:
        """
        Retrieve global configuration.

        :return: Dictionary containing global configurations.
        """
        return self._global_config

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Retrieve logging configuration.

        :return: Dictionary containing logging configurations.
        """
        return self._logging_config

    def get_storage_config(self) -> Dict[str, Any]:
        """
        Retrieve storage configuration.

        :return: Dictionary containing storage configurations.
        """
        return self._storage_config

    def get_hessian_config(self) -> Dict[str, Any]:
        """
        Retrieve Hessian configuration.

        :return: Dictionary containing Hessian configurations.
        """
        return self._hessian_config

    def get_analysis_config(self) -> Dict[str, Any]:
        """
        Retrieve analysis configuration.

        :return: Dictionary containing analysis configurations.
        """
        return self._analysis_config

    def get_lora_config(self) -> Dict[str, Any]:
        """
        Retrieve LoRA configuration.

        :return: Dictionary containing LoRA configurations.
        """
        return self._lora_config

    def _set_log_dir(self) -> None:
        """
        Set single logging directory for all components.
        """
        log_root = self._global_config.get("log_root", "./analog")
        log_dir = os.path.join(log_root, self.project_name)

        self._global_config["log_dir"] = log_dir
        self._logging_config["log_dir"] = log_dir
        self._storage_config["log_dir"] = log_dir
        self._hessian_config["log_dir"] = log_dir + "/hessian"
