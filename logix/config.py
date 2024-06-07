# Copyright 2023-present the LogIX team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import yaml

from logix.utils import get_rank


def init_config_from_yaml(project: str, logix_config: Optional[str] = None):
    config_dict = {}
    if logix_config is not None:
        assert os.path.exists(logix_config), f"{logix_config} doesn't exist!"
        with open(logix_config, "r", encoding="utf-8") as config_file:
            config_dict = yaml.safe_load(config_file)

    assert project is not None
    config_dict["project"] = project

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
        default=1000000000, metadata={"help": "Flush threshold for the log buffer."}
    )
    num_workers: int = field(
        default=1, metadata={"help": "Number of workers used for logging."}
    )
    cpu_offload: int = field(
        default=False, metadata={"help": "Offload statistic states to CPU."}
    )
    log_dtype: str = field(default="none", metadata={"help": "Data type for logging."})

    def get_dtype(self):
        if self.log_dtype == "none":
            return None
        elif self.log_dtype == "float64":
            return torch.float64
        elif self.log_dtype == "float16":
            return torch.float16
        elif self.log_dtype == "bfloat16":
            return torch.bfloat16
        elif self.log_dtype == "int8":
            return torch.int8
        return torch.float32


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
        default=1e-10, metadata={"help": "Damping strength for influence."}
    )
    relative_damping: bool = field(
        default=False,
        metadata={"help": "Compute the damping term based on sigular values."},
    )
    mode: str = field(default="dot", metadata={"help": "Mode for influence."})
    flatten: bool = field(
        default=False, metadata={"help": "flattening flag for logging"}
    )


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
        default="./logix", metadata={"help": "Root directory for logging."}
    )
    logging: Union[Dict[str, Any], LoggingConfig] = field(
        default_factory=LoggingConfig, metadata={"help": "Logging configuration."}
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
        self.logging.log_dir = self.log_dir

        if not os.path.exists(self.log_dir) and get_rank() == 0:
            os.makedirs(self.log_dir)

    def load_config(self, logix_config: str) -> None:
        """
        Load configuration from the saved YAML file.

        Args:
            logix_config: Path to the saved YAML file.
        """
        assert os.path.exists(logix_config), "Configuration file not found."
        config_dict = {}
        with open(logix_config, "r", encoding="utf-8") as config_file:
            config_dict = yaml.safe_load(config_file)

        load_config_from_dict(self, config_dict)
        self.configure_log_dir()

    def save_config(self, log_dir: Optional[str] = None) -> None:
        """
        Save configuration to a YAML file.
        """
        if get_rank() == 0:
            config_file = os.path.join(log_dir or self.log_dir, "config.yaml")
            config_dict = asdict(self)

            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
