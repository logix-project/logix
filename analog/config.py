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
        "global": {},
        "logging": {},
        "storage": {"type": "default", "log_dir": "./analog"},
        "hessian": {"type": "kfac", "damping": 1e-2},
        "analysis": {},
        "lora": {"init": "pca", "rank": 64},
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
            get_logger().warning(
                "Configuration file not found. Using default values.\n"
            )
            self.data = {}

        self._set_global_config()

    def get_global_config(self) -> Dict[str, Any]:
        """
        Retrieve global configuration.

        :return: Dictionary containing global configurations.
        """
        return self.data.get("global", self._DEFAULTS["global"])

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Retrieve logging configuration.

        :return: Dictionary containing logging configurations.
        """
        return self.data.get("logging", self._DEFAULTS["logging"])

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

    def get_analysis_config(self) -> Dict[str, Any]:
        """
        Retrieve analysis configuration.

        :return: Dictionary containing analysis configurations.
        """
        return self.data.get("analysis", self._DEFAULTS["analysis"])

    def get_lora_config(self) -> Dict[str, Any]:
        """
        Retrieve LoRA configuration.

        :return: Dictionary containing LoRA configurations.
        """
        return self.data.get("lora", self._DEFAULTS["lora"])

    def _set_global_config(self) -> None:
        """
        Set global configurations.
        """
        # There should be just 1 disk location ("log_dir")
        global_log_dir = self.get_global_config().get("log_dir")
        storage_log_dir = self.get_storage_config().get("log_dir")
        if global_log_dir is not None and storage_log_dir is not None:
            if global_log_dir != storage_log_dir:
                raise ValueError(
                    f"Global log directory and storage log directory clash: "
                    f"global={global_log_dir}, storage={storage_log_dir}"
                )
        # one should be None, or both should be the same
        assert global_log_dir or storage_log_dir, "No log directory specified"
        log_dir = global_log_dir or storage_log_dir
        # Setting global log directory
        if len(self.data) == 0:
            self._DEFAULTS["global"]["log_dir"] = log_dir
        else:
            self.data["global"]["log_dir"] = log_dir
