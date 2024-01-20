import os

from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field, is_dataclass
import yaml

from analog.utils import get_rank


def init_config_from_yaml(config_path: str, project: Optional[str] = None):
    config_dict = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as config_file:
            config_dict = yaml.safe_load(config_file)

    if project is not None:
        config_dict["project"] = project
    assert "project" in config_dict, "Project name must be provided."

    return Config(**config_dict)


def load_config_from_dict(config, config_dict: Dict[str, Any]):
    for name, value in config_dict.items():
        if isinstance(value, dict):
            assert is_dataclass(getattr(config, name))
            load_config_from_dict(getattr(config, name), value)
        else:
            setattr(config, name, value)


@dataclass
class LoggingConfig:
    """
    Configuration for logging.

    Args:
        flush_threshold: Flush threshold for the buffer.
        num_workers: Number of workers used for log saving.
        cpu_offload: Offload statistic states to CPU.
    """

    log_dir: str = field(init=False)
    flush_threshold: int = field(
        default=-1, metadata={"help": "Flush threshold for the log buffer."}
    )
    num_workers: int = field(
        default=1, metadata={"help": "Number of workers used for logging."}
    )
    cpu_offload: int = field(
        default=False, metadata={"help": "Offload statistic states to CPU."}
    )


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA.

    Args:
        init: Initialization method for LoRA.
        rank: Rank for LoRA.
        parameter_sharing: Parameter sharing for LoRA.
        parameter_sharing_groups: Parameter sharing groups for LoRA.
    """

    init: str = field(
        default="random", metadata={"help": "Initialization method for LoRA."}
    )
    rank: int = field(default=64, metadata={"help": "Rank for LoRA."})
    parameter_sharing: bool = field(
        default=False, metadata={"help": "Parameter sharing for LoRA."}
    )
    parameter_sharing_groups: Optional[List[str]] = field(
        default=None, metadata={"help": "Parameter sharing groups for LoRA."}
    )


@dataclass
class InfluenceConfig:
    """
    Configuration for influence.

    Args:
        damping: Damping for influence.
        relative_damping: Compute the damping term based on sigular values.
        mode: Mode for influence.
    """

    log_dir: str = field(init=False)
    damping: float = field(
        default=0.0, metadata={"help": "Damping strength for influence."}
    )
    relative_damping: bool = field(
        default=False,
        metadata={"help": "Compute the damping term based on sigular values."},
    )
    mode: str = field(default="dot", metadata={"help": "Mode for influence."})


@dataclass
class Config:
    """
    Configuration management class.
    Loads configurations from a YAML file and provides access to component-specific configurations.

    Args:
        project: Project name.
        root_dir: Root directory for logging.
        logging: Logging configuration.
        lora: LoRA configuration.
        analysis: Analysis configuration.
    """

    project: str
    log_dir: str = field(init=False)
    root_dir: str = field(
        default="./analog", metadata={"help": "Root directory for logging."}
    )
    logging: Union[Dict[str, Any], LoggingConfig] = field(
        default_factory=LoggingConfig, metadata={"help": "Logging configuration."}
    )
    influence: Union[Dict[str, Any], InfluenceConfig] = field(
        default_factory=InfluenceConfig, metadata={"help": "Influence configuration."}
    )
    lora: Union[Dict[str, Any], LoRAConfig] = field(
        default_factory=LoRAConfig, metadata={"help": "LoRA configuration."}
    )

    def __post_init__(self):
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)
        if isinstance(self.lora, dict):
            self.lora = LoRAConfig(**self.lora)

        self.log_dir = None
        self.configure_log_dir()

    def configure_log_dir(self) -> None:
        """
        Set single logging directory for all components.
        """
        self.log_dir = os.path.join(self.root_dir, self.project)

        if not os.path.exists(self.log_dir) and get_rank() == 0:
            os.makedirs(self.log_dir)

        self.logging.log_dir = self.log_dir
        self.influence.log_dir = self.log_dir

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from the saved YAML file.

        Args:
            config_path: Path to the saved YAML file.
        """
        assert os.path.exists(config_path), "Configuration file not found."
        config_dict = {}
        with open(config_path, "r", encoding="utf-8") as config_file:
            config_dict = yaml.safe_load(config_file)

        load_config_from_dict(self, config_dict)
        self.configure_log_dir()
