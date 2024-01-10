import os

from typing import Optional, Iterable, Dict, Any, List

import torch
import torch.nn as nn

from analog.analysis import AnalysisBase
from analog.batch_info import BatchInfo
from analog.config import Config
from analog.logging import HookLogger
from analog.logging.log_loader import LogDataset
from analog.logging.log_loader_util import collate_nested_dicts
from analog.lora import LoRAHandler
from analog.lora.utils import is_lora
from analog.state import StatisticState
from analog.utils import (
    get_logger,
    get_rank,
    get_world_size,
    print_tracked_modules,
    module_check,
)


class AnaLog:
    """
    AnaLog is a tool for logging and analyzing neural networks.
    """

    _SUPPORTED_MODULES = {nn.Linear, nn.Conv1d, nn.Conv2d}

    def __init__(
        self,
        project: str,
        config: str = "",
    ) -> None:
        """
        Initializes the AnaLog class for neural network logging.

        Args:
            project (str): The name or identifier of the project.
            config (str, optional): The path to the YAML configuration file. Defaults to "".
        """
        self.project = project

        self.model = None

        # Config
        self.config = Config(config_file=config, project_name=project)
        self.logging_config = self.config.logging_config
        self.lora_config = self.config.lora_config
        self.log_dir = self.config.log_dir

        # AnaLog state
        self.state = StatisticState()
        self.binfo = BatchInfo()

        # Initialize logger
        self.logger = HookLogger(
            config=self.logging_config, state=self.state, binfo=self.binfo
        )

        # Analysis plugins
        self.analysis_plugins = {}

        self.type_filter = None
        self.name_filter = None

    def watch(
        self,
        model: nn.Module,
        type_filter: List[nn.Module] = None,
        name_filter: List[str] = None,
    ) -> None:
        """
        Sets up modules in the model to be watched.

        Args:
            model: The neural network model.
            type_filter (list, optional): List of types of modules to be watched.
            name_filter (list, optional): List of keyword names for modules to be watched.
            lora (bool, optional): Whether to use LoRA to watch the model.
        """
        self.model = model
        if get_world_size() > 1 and hasattr(self.model, "module"):
            self.model = self.model.module
        self.type_filter = type_filter or self.type_filter
        self.name_filter = name_filter or self.name_filter

        _is_lora = is_lora(self.model)

        for name, module in self.model.named_modules():
            if module_check(
                module=module,
                module_name=name,
                supported_modules=self._SUPPORTED_MODULES,
                type_filter=self.type_filter,
                name_filter=self.name_filter,
                is_lora=_is_lora,
            ):
                self.logger.add_module(name, module)
        print_tracked_modules(self.logger.modules_to_name)

        self.logger.register_all_module_hooks()

    def watch_activation(self, tensor_dict: Dict[str, torch.Tensor]) -> None:
        """
        Sets up tensors to be watched.

        Args:
            tensor_dict (dict): Dictionary containing tensor names as keys and tensors as values.
        """
        self.logger.register_all_tensor_hooks(tensor_dict)

    def add_lora(
        self,
        model: Optional[nn.Module] = None,
        watch: bool = True,
        clear: bool = True,
        lora_state: Dict[str, Any] = None,
    ) -> None:
        """
        Adds LoRA for gradient compression.

        Args:
            model: The neural network model.
            parameter_sharing (bool, optional): Whether to use parameter sharing or not.
            parameter_sharing_groups (list, optional): List of parameter sharing groups.
            watch (bool, optional): Whether to watch the model or not.
            clear (bool, optional): Whether to clear the internal states or not.
        """
        if not hasattr(self, "lora_handler"):
            self.lora_handler = LoRAHandler(self.lora_config, self.state)
            self.lora_config = self.config.lora_config

        if model is None:
            model = self.model

        self.lora_handler.add_lora(
            model=model,
            type_filter=self.type_filter,
            name_filter=self.name_filter,
            lora_state=lora_state,
        )

        # Clear state and logger
        if clear:
            msg = "AnaLog will clear the previous Hessian, Storage, and Logging "
            msg += "handlers after adding LoRA for gradient compression.\n"
            get_logger().info(msg)
            self.clear()
        if watch:
            self.watch(model)

    def add_analysis(self, analysis_dict: Dict[str, AnalysisBase]) -> None:
        """
        Adds analysis plugins to AnaLog.

        Args:
            analysis_dict (dict): Dictionary containing analysis names as keys and analysis classes as values.
        """
        for analysis_name, analysis_cls in analysis_dict.items():
            if hasattr(self, analysis_name):
                raise ValueError(f"Analysis name {analysis_name} is reserved.")
            analysis_plugin = analysis_cls(self.config.analysis_config, self.state)
            setattr(
                self,
                analysis_name,
                analysis_plugin,
            )
            self.analysis_plugins[analysis_name] = getattr(self, analysis_name)

        return analysis_plugin

    def remove_analysis(self, analysis_name: str) -> None:
        """
        Removes analysis plugins from AnaLog.

        Args:
            analysis_name (str): Name of the analysis to be removed.
        """
        if analysis_name not in self.analysis_plugins:
            get_logger().warning(
                f"Analysis {analysis_name} does not exist. Nothing to remove."
            )
            return None
        del self.analysis_plugins[analysis_name]
        delattr(self, analysis_name)

    def log(self, data_id: Any, mask: Optional[torch.Tensor] = None):
        """
        Logs the data.

        Args:
            data_id: A unique identifier associated with the data for the logging session.
            mask (torch.Tensor, optional): Mask for the data.
        """
        self.logger.log(data_id=data_id, mask=mask)

    def __call__(
        self,
        data_id: Iterable[Any] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            data_id: A unique identifier associated with the data for the logging session.
            mask (torch.Tensor, optional): Mask for the data.

        Returns:
            self: Returns the instance of the AnaLog object.
        """
        self.binfo.clear()
        self.binfo.data_id = data_id
        self.binfo.mask = mask

        return self

    def __enter__(self):
        """
        Sets up the context manager.

        This method is automatically called when the `with` statement is used with an `AnaLog` object.
        It sets up the logging environment based on the provided parameters.
        """
        self.logger.clear(hook=True, module=False, buffer=False)
        self.logger.register_all_module_hooks()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Clears the internal states and removes all the hooks.

        This method is essential for ensuring that there are no lingering hooks that could
        interfere with further operations on the model or with future logging sessions.
        """
        self.logger.update()

    def build_log_dataset(self):
        """
        Constructs the log dataset from the storage handler.
        """
        return LogDataset(log_dir=self.log_dir)

    def build_log_dataloader(
        self, batch_size: int = 16, num_workers: int = 0, pin_memory: bool = False
    ):
        """
        Constructs the log dataloader from the storage handler.
        """
        log_dataset = self.build_log_dataset()
        log_dataloader = torch.utils.data.DataLoader(
            log_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=collate_nested_dicts,
        )
        return log_dataloader

    def get_log(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns the current log.

        Returns:
            dict: The current log.
        """
        return self.binfo.data_id, self.binfo.log

    def get_covariance_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns the covariance state from the Hessian handler.
        """
        return self.state.get_covariance_state()

    def get_covariance_svd_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns the SVD of the covariance from the Hessian handler.
        """
        return self.state.get_covariance_svd_state()

    def save_config(self) -> None:
        """
        Save AnaLog state to disk.
        """
        torch.save(self.config, os.path.join(self.log_dir, "config.pt"))

    def save_state(self) -> None:
        """
        Save Hessian state to disk.
        """
        self.state.save_state(self.log_dir)

    def save_lora(self) -> None:
        """
        Save LoRA state to disk.
        """
        state_dict = self.model.state_dict()
        lora_state_dict = {
            name: param for name, param in state_dict.items() if "analog_lora" in name
        }
        if len(lora_state_dict) > 0:
            log_dir = os.path.join(self.log_dir, "lora")
            if not os.path.exists(log_dir) and get_rank() == 0:
                os.makedirs(log_dir)
            torch.save(lora_state_dict, os.path.join(log_dir, "lora_state_dict.pt"))

    def initialize_from_log(self) -> None:
        """
        Load all states from disk.
        """
        # Load analog config
        assert os.path.exists(self.log_dir), f"{self.log_dir} does not exist!"

        config_path = os.path.join(self.log_dir, "config.pt")
        self.config.load_config(config_path)

        # Load LoRA state
        lora_dir = os.path.join(self.log_dir, "lora")
        if os.path.exists(lora_dir):
            lora_state = torch.load(os.path.join(lora_dir, "lora_state_dict.pt"))
            if not is_lora(self.model):
                self.add_lora(lora_state=lora_state)
            for name in lora_state:
                assert name in self.model.state_dict(), f"{name} not in model!"
            self.model.load_state_dict(lora_state, strict=False)

        # Load state
        self.state.load_state(self.log_dir)

    def finalize(
        self,
    ) -> None:
        """
        Finalizes the logging session.

        Args:
            clear (bool, optional): Whether to clear the internal states or not.
        """
        self.state.finalize()
        self.logger.finalize()

        if get_rank() == 0:
            self.save_config()
            self.save_state()
            self.save_lora()

    def setup(self, log_option_kwargs: Dict[str, Any]) -> None:
        """
        Update logging configurations.

        Args:
            log_option_kwargs: Logging configurations.
        """
        self.logger.opt.setup(log_option_kwargs)

    def eval(self) -> None:
        """
        Set the state of AnaLog for testing.
        """
        self.logger.opt.eval()

    def clear(self) -> None:
        """
        Clear everything in AnaLog.
        """
        self.state.clear()
        self.logger.clear()
        for key in self.analysis_plugins:
            self.remove_analysis(key)
