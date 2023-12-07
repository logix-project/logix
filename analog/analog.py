import os

from typing import Optional, Iterable, Dict, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn

from analog.config import Config
from analog.constants import FORWARD, BACKWARD, GRAD, LOG_TYPES
from analog.logging import LoggingHandler
from analog.storage import DefaultStorageHandler
from analog.hessian import RawHessianHandler, KFACHessianHandler
from analog.analysis import AnalysisBase
from analog.lora import LoRAHandler
from analog.utils import get_logger, get_rank


class AnaLogState:
    def __init__(self) -> None:
        self.log = []
        self.hessian = False
        self.save = False
        self.test = False

    def set_state(
        self,
        state_kwargs: Dict[str, Any],
    ) -> None:
        for key, value in state_kwargs.items():
            assert hasattr(self, key), f"Invalid state key: {key}"
            setattr(self, key, value)


class AnaLog:
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
        config = Config(config_file=config, project_name=project)
        self.config = config

        # Log dir
        self.log_dir = config.get_log_dir()

        # Initialize storage, hessian, and logging handlers from config as well as
        # inject dependencies between handlers.
        self.storage_handler = self.build_storage_handler()
        self.hessian_handler = self.build_hessian_handler()
        self.logging_handler = self.build_logging_handler()
        self.lora_handler = self.build_lora_handler()

        # Analysis plugins
        self.analysis_plugins = {}

        # Internal states
        self.state = AnaLogState()

        self.type_filter = None
        self.name_filter = None

    def watch(
        self,
        model: nn.Module,
        type_filter: List[nn.Module] = None,
        name_filter: List[str] = None,
        lora: bool = False,
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
        self.type_filter = type_filter or self.type_filter
        self.name_filter = name_filter or self.name_filter

        for name, module in model.named_modules():
            # only consider the leaf module
            if len(list(module.children())) > 0:
                continue
            if not any(
                isinstance(module, module_type)
                for module_type in self._SUPPORTED_MODULES
            ):
                continue
            if type_filter is not None and not any(
                isinstance(module, module_type) for module_type in self.type_filter
            ):
                continue
            if name_filter is not None and not any(
                keyword in name for keyword in self.name_filter
            ):
                continue
            if lora and "analog_lora_B" not in name:
                continue
            self.logging_handler.add_module(name, module)
        self.print_tracked_modules()

    def watch_activation(self, tensor_dict: Dict[str, torch.Tensor]) -> None:
        """
        Sets up tensors to be watched.

        Args:
            tensor_dict (dict): Dictionary containing tensor names as keys and tensors as values.
        """
        self.logging_handler.register_all_tensor_hooks(tensor_dict)

    def add_lora(
        self,
        model: Optional[nn.Module] = None,
        parameter_sharing: bool = False,
        parameter_sharing_groups: List[str] = None,
        watch: bool = True,
        clear: bool = True,
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
        if model is None:
            model = self.model

        hessian_state = self.hessian_handler.get_hessian_state()
        self.lora_handler.add_lora(
            model=model,
            type_filter=self.type_filter,
            name_filter=self.name_filter,
            hessian_state=hessian_state,
            parameter_sharing=parameter_sharing,
            parameter_sharing_groups=parameter_sharing_groups,
        )

        # Clear hessian, storage, and logging handlers
        if clear:
            msg = "AnaLog will clear the previous Hessian, Storage, and Logging "
            msg += "handlers after adding LoRA for gradient compression.\n"
            get_logger().info(msg)
            self.clear()
        if watch:
            self.watch(model, lora=True)

    def add_analysis(self, analysis_dict: Dict[str, AnalysisBase]) -> None:
        """
        Adds analysis plugins to AnaLog.

        Args:
            analysis_dict (dict): Dictionary containing analysis names as keys and analysis classes as values.
        """
        for analysis_name, analysis_cls in analysis_dict.items():
            if hasattr(self, analysis_name):
                raise ValueError(f"Analysis name {analysis_name} is reserved.")
            setattr(
                self,
                analysis_name,
                analysis_cls(
                    self.config.get_analysis_config(),
                    self.storage_handler,
                    self.hessian_handler,
                ),
            )
            self.analysis_plugins[analysis_name] = getattr(self, analysis_name)

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

    def __call__(
        self,
        data_id: Optional[Iterable[Any]] = None,
        log: Optional[Iterable[str]] = None,
        hessian: Optional[bool] = None,
        save: Optional[bool] = None,
        test: bool = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            data_id: A unique identifier associated with the data for the logging session.
            log (str, optional): Specifies which data to log (e.g. "gradient", "full_activations").
            hessian (bool, optional): Whether to compute the Hessian or not.
            save (bool, optional): Whether to save the logs or not.
            test (bool, optional): Whether the logging is for the test phase or not.

        Returns:
            self: Returns the instance of the AnaLog object.
        """
        state = {}
        if log is not None:
            state["log"] = log
        if hessian is not None:
            state["hessian"] = hessian
        if save is not None:
            state["save"] = save
        if test:
            state["test"] = test
        self.set_state(state)

        self.data_id = data_id
        self.mask = mask

        return self

    def __enter__(self):
        """
        Sets up the context manager.

        This method is automatically called when the `with` statement is used with an `AnaLog` object.
        It sets up the logging environment based on the provided parameters.
        """

        self.storage_handler.set_data_id(self.data_id)

        self.logging_handler.set_states(self.state)
        self.logging_handler.set_mask(self.mask)
        self.logging_handler.register_all_module_hooks()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Clears the internal states and removes all the hooks.

        This method is essential for ensuring that there are no lingering hooks that could
        interfere with further operations on the model or with future logging sessions.
        """

        # Wait for all async operations to finish
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()

        # Compute the Hessian if necessary
        if self.state.hessian:
            self.hessian_handler.on_exit(self.get_log())

        # Flush the storage handler if necessary
        if self.state.save:
            self.storage_handler.flush()

        # Remove all hooks
        self.logging_handler.clear()

    def build_storage_handler(self) -> None:
        """
        Initialize a storage handler from the configuration.

        Returns:
            The initialized storage handler.
        """
        storage_config = self.config.get_storage_config()
        storage_type = storage_config.get("type", "default")
        if storage_type == "default":
            return DefaultStorageHandler(storage_config)
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

    def build_hessian_handler(self):
        """
        Initialize a Hessian handler from the configuration.

        Returns:
            The initialized Hessian handler.
        """
        hessian_config = self.config.get_hessian_config()
        hessian_type = hessian_config.get("type", "kfac")
        if hessian_type == "kfac":
            return KFACHessianHandler(hessian_config)
        elif hessian_type == "raw":
            return RawHessianHandler(hessian_config)
        else:
            raise ValueError(f"Unknown Hessian type: {hessian_type}")

    def build_logging_handler(self):
        """
        Initialize a Logging handler from the configuration.

        Returns:
            The initialized Hessian handler.
        """
        logging_config = self.config.get_logging_config()
        return LoggingHandler(
            logging_config, self.storage_handler, self.hessian_handler
        )

    def build_lora_handler(self):
        """
        Initialize a Logging handler from the configuration.

        Returns:
            The initialized Hessian handler.
        """
        lora_config = self.config.get_lora_config()
        return LoRAHandler(lora_config)

    def build_log_dataset(self):
        """
        Constructs the log dataset from the storage handler.
        """
        return self.storage_handler.build_log_dataset()

    def build_log_dataloader(self, batch_size=16, num_workers=0):
        """
        Constructs the log dataloader from the storage handler.
        """
        return self.storage_handler.build_log_dataloader(batch_size, num_workers)

    def get_log(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns the current log.

        Returns:
            dict: The current log.
        """
        return self.logging_handler.current_log

    def get_storage_buffer(self):
        """
        Returns the storage buffer from the storage handler.
        """
        return self.storage_handler.get_buffer()

    def get_hessian_state(
        self, copy: bool = False
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns the Hessian state from the Hessian handler.
        """
        hessian_state = self.hessian_handler.get_hessian_state()
        if copy:
            return hessian_state.copy()
        return hessian_state

    def get_hessian_svd_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns the SVD of the Hessian from the Hessian handler.
        """
        return self.hessian_handler.get_hessian_svd_state()

    def hessian_inverse(self):
        """
        Compute the inverse of the Hessian.
        """
        return self.hessian_handler.hessian_inverse()

    def hessian_svd(self):
        """
        Compute the SVD of the Hessian.
        """
        return self.hessian_handler.hessian_svd()

    def save_analog_state(self) -> None:
        """
        Save AnaLog state to disk.
        """
        torch.save(self.state, os.path.join(self.log_dir, "analog_state.pt"))

    def save_hessian_state(self) -> None:
        """
        Save Hessian state to disk.
        """
        self.hessian_handler.save_state()

    def save_lora_state(self) -> None:
        """
        Save LoRA state to disk.
        """
        state_dict = self.model.state_dict()
        lora_state_dict = {
            name: param for name, param in state_dict.items() if "analog_lora" in name
        }
        if len(lora_state_dict) > 0:
            log_dir = os.path.join(self.log_dir, "lora")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            torch.save(lora_state_dict, os.path.join(log_dir, "lora_state.pt"))

    def initialize_from_log(self) -> None:
        """
        Load all states from disk.
        """
        # Load analog state
        analog_state_path = os.path.join(self.log_dir, "analog_state.pt")
        if os.path.exists(analog_state_path):
            self.state = torch.load(analog_state_path)

        # Load hessian state
        hessian_dir = os.path.join(self.log_dir, "hessian")
        if os.path.exists(hessian_dir):
            self.hessian_handler.load_state(hessian_dir)

        # Load LoRA state
        lora_dir = os.path.join(self.log_dir, "lora")
        if os.path.exists(lora_dir):
            lora_state = torch.load(os.path.join(lora_dir, "lora_state.pt"))
            for name in lora_state:
                assert name in self.model.state_dict(), f"{name} not in model!"
            self.model.load_state_dict(lora_state, strict=False)

    def finalize(
        self,
        clear: bool = False,
    ) -> None:
        """
        Finalizes the logging session.

        Args:
            clear (bool, optional): Whether to clear the internal states or not.
        """
        self.hessian_handler.finalize()
        self.storage_handler.finalize()

        if get_rank() == 0:
            self.save_analog_state()
            self.save_hessian_state()
            self.save_lora_state()

        if clear:
            self.hessian_handler.clear()
            self.storage_handler.clear()

    def sanity_check(self) -> None:
        """
        Performs a sanity check on the provided parameters.
        """
        state = self.state
        if len(state.log) > 0 and len(set(state.log) - LOG_TYPES) > 0:
            raise ValueError("Invalid value for 'log'.")
        if state.test and (state.hessian or state.save):
            get_logger().warning(
                "Cannot compute Hessian or save logs during testing. "
                + "Setting 'hessian' and 'save' to False."
            )
            state.hessian = False
            state.save = False
        if GRAD in state.log and len(state.log) > 1:
            raise ValueError("Cannot log 'grad' with other log types.")

    def ekfac(self, on: bool = True) -> None:
        """
        Compute the EKFAC approximation of the Hessian.
        """
        assert self.hessian_handler.config.get("type", "kfac") == "kfac"
        if on:
            self.hessian_handler.ekfac = True
        else:
            self.hessian_handler.ekfac = False

    def set_state(self, state_kwargs: Dict[str, Any]) -> None:
        """
        Set the state of AnaLog.

        Args:
            state_kwargs (Dict[str, Any]): The state to be set.
        """
        self.state.set_state(state_kwargs)
        self.sanity_check()

    def eval(self) -> None:
        """
        Set the state of AnaLog for testing.
        """
        state = {"hessian": False, "save": False, "test": True}
        self.state.set_state(state)

    def clear(self) -> None:
        """
        Clear everything in AnaLog.
        """
        self.logging_handler.clear(clear_modules=True)
        self.storage_handler.clear()
        self.hessian_handler.clear()
        for key in self.analysis_plugins:
            self.remove_analysis(key)

    def print_tracked_modules(self) -> None:
        """
        Print the tracked modules.
        """
        get_logger().info("Tracking the following modules:")
        repr_dim = 0
        for k, v in self.logging_handler.modules_to_name.items():
            get_logger().info(f"{v}: {k}")
            repr_dim += k.weight.data.numel()
        get_logger().info(f"Total number of parameters: {repr_dim:,}\n")