import os
from typing import Dict, Any
import yaml

import torch

from analog.utils import get_logger, get_rank


class Config:
    """
    Configuration management class.
    Loads configurations from a YAML file and provides access to component-specific configurations.
    """

    # Default values for each configuration
    _DEFAULTS = {
        "root_dir": "./analog",
        "logging": {"flush_threshold": -1},
        "analysis": {},
        "lora": {"init": "random", "rank": 64},
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

        self._logging_config = self.data.get("logging", self._DEFAULTS["logging"])
        self._analysis_config = self.data.get("analysis", self._DEFAULTS["analysis"])
        self._lora_config = self.data.get("lora", self._DEFAULTS["lora"])

        self._log_dir = None
        self._configure_log_dir(self.data.get("root_dir", self._DEFAULTS["root_dir"]))

    @property
    def logging_config(self) -> Dict[str, Any]:
        """
        Retrieve logging configuration.

        :return: Dictionary containing logging configurations.
        """
        return self._logging_config

    @property
    def analysis_config(self) -> Dict[str, Any]:
        """
        Retrieve analysis configuration.

        :return: Dictionary containing analysis configurations.
        """
        return self._analysis_config

    @property
    def lora_config(self) -> Dict[str, Any]:
        """
        Retrieve LoRA configuration.

        :return: Dictionary containing LoRA configurations.
        """
        return self._lora_config

    @property
    def log_dir(self) -> str:
        """
        Retrieve logging directory.

        :return: Path to logging directory.
        """
        return self._log_dir

    def _configure_log_dir(self, root_dir) -> None:
        """
        Set single logging directory for all components.
        """
        self._log_dir = os.path.join(root_dir, self.project_name)

        if not os.path.exists(self._log_dir) and get_rank() == 0:
            os.makedirs(self._log_dir)

        self._logging_config["log_dir"] = self._log_dir

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from the saved YAML file.

        Args:
            config_path: Path to the saved YAML file.
        """
        config = torch.load(config_path)
        self._logging_config.update(config.logging_config)
        self._analysis_config.update(config.analysis_config)
        self._lora_config.update(config.lora_config)
        self._log_dir = config.log_dir
