import os

from typing import Optional, Iterable, Dict, Any, List

import torch
import torch.nn as nn

from analog.analysis import AnalysisBase
from analog.config import Config
from analog.constants import GRAD, LOG_TYPES
from analog.logging import LoggingHandler
from analog.state import AnaLogState
from analog.storage import StorageHandler
from analog.hessian import RawHessianHandler, KFACHessianHandler
from analog.lora import LoRAHandler
from analog.storage.log_saver import LogSaver
from analog.utils import get_logger, get_rank, get_world_size


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
        self.logging_config = self.config.get_logging_config()
        self.hessian_config = self.config.get_hessian_config()
        self.storage_config = self.config.get_storage_config()
        self.lora_config = self.config.get_lora_config()
        self.log_dir = self.config.get_log_dir()

        # AnaLog state
        self.state = AnaLogState()

        # Initialize storage, hessian, and logging handlers from config as well as
        # inject dependencies between handlers.
        self.storage_handler = self.build_storage_handler(
            storage_config=self.storage_config, state=self.state
        )
        self.hessian_handler = self.build_hessian_handler(
            hessian_config=self.hessian_config, state=self.state
        )
        self.logging_handler = self.build_logging_handler(
            logging_config=self.logging_config,
            state=self.state,
            hessian_handler=self.hessian_handler,
        )
        self.lora_handler = self.build_lora_handler(
            lora_config=self.lora_config, state=self.state
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
        if get_world_size() > 1 and hasattr(self.model, "module"):
            self.model = self.model.module
        self.type_filter = type_filter or self.type_filter
        self.name_filter = name_filter or self.name_filter

        for name, module in self.model.named_modules():
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

        self.lora_handler.add_lora(
            model=model,
            type_filter=self.type_filter,
            name_filter=self.name_filter,
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
                analysis_cls(self.config.get_analysis_config(), self.state),
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
        logging_config = {}
        if log is not None:
            logging_config["log"] = log
        if hessian is not None:
            logging_config["hessian"] = hessian
        if save is not None:
            logging_config["save"] = save
        if test:
            logging_config["test"] = test
        self.update(logging_config)

        self.data_id = data_id
        self.mask = mask

        return self

    def __enter__(self):
        """
        Sets up the context manager.

        This method is automatically called when the `with` statement is used with an `AnaLog` object.
        It sets up the logging environment based on the provided parameters.
        """

        self.state.clear_log_state()
        self.storage_handler.set_data_id(self.data_id)
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
        if self.logging_config["hessian"]:
            self.hessian_handler.on_exit(self.get_log())

        # Flush the storage handler if necessary
        if self.logging_config["save"]:
            self.storage_handler.buffer_write_on_exit()
            self.storage_handler.flush()

        # Remove all hooks
        self.logging_handler.clear()

    def build_storage_handler(self, storage_config, state) -> None:
        """
        Initialize a storage handler from the configuration.

        Returns:
            The initialized storage handler.
        """
        log_saver = LogSaver()
        return StorageHandler(log_saver, storage_config, state)

    def build_hessian_handler(self, hessian_config, state):
        """
        Initialize a Hessian handler from the configuration.

        Returns:
            The initialized Hessian handler.
        """
        hessian_type = hessian_config.get("type", "kfac")
        if hessian_type == "kfac":
            return KFACHessianHandler(hessian_config, state)
        elif hessian_type == "raw":
            return RawHessianHandler(hessian_config, state)
        else:
            raise ValueError(f"Unknown Hessian type: {hessian_type}")

    def build_logging_handler(self, logging_config, state, hessian_handler):
        """
        Initialize a Logging handler from the configuration.

        Returns:
            The initialized Hessian handler.
        """
        return LoggingHandler(logging_config, state, hessian_handler)

    def build_lora_handler(self, lora_config, state):
        """
        Initialize a Logging handler from the configuration.

        Returns:
            The initialized Hessian handler.
        """
        return LoRAHandler(lora_config, state)

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
        return self.state.log_state

    def get_hessian_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns the Hessian state from the Hessian handler.
        """
        return self.state.get_hessian_state()

    def get_hessian_svd_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns the SVD of the Hessian from the Hessian handler.
        """
        return self.state.get_hessian_svd_state()

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

        # Load state
        self.state.load_state(self.log_dir)

        # Load LoRA state
        lora_dir = os.path.join(self.log_dir, "lora")
        if os.path.exists(lora_dir):
            lora_state = torch.load(os.path.join(lora_dir, "lora_state_dict.pt"))
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
        self.state.finalize()
        self.storage_handler.finalize()

        if get_rank() == 0:
            self.save_config()
            self.save_state()
            self.save_lora()

        if clear:
            # TODO: should we clear `state`?
            self.state.clear()
            self.storage_handler.clear()

    def sanity_check(self) -> None:
        """
        Performs a sanity check on the provided parameters.
        """
        config = self.logging_config
        if len(config["log"]) > 0 and len(set(config["log"]) - LOG_TYPES) > 0:
            raise ValueError("Invalid value for 'log'.")
        if GRAD in config["log"] and len(config["log"]) > 1:
            raise ValueError("Cannot log 'grad' with other log types.")

    def ekfac(self, on: bool = True) -> None:
        """
        Compute the EKFAC approximation of the Hessian.
        """
        assert self.hessian_handler.config.get("type", "kfac") == "kfac"
        if on:
            get_logger().info("Enabling EKFAC approximation for the Hessian.\n")
            self.state.hessian_svd()
            self.state.register_state("ekfac_eigval_state", synchronize=True, save=True)
            self.state.register_state("ekfac_counter", synchronize=True, save=False)
            self.state.register_normalize_pair("ekfac_eigval_state", "ekfac_counter")

            self.hessian_handler.ekfac = True
        else:
            get_logger().info("Disabling EKFAC approximation for the Hessian.\n")
            self.hessian_handler.ekfac = False

    def update(self, logging_config_kwargs: Dict[str, Any]) -> None:
        """
        Set the state of AnaLog.

        Args:
            state_kwargs (Dict[str, Any]): The state to be set.
        """
        self.logging_config.update(logging_config_kwargs)
        self.sanity_check()

    def eval(self) -> None:
        """
        Set the state of AnaLog for testing.
        """
        eval_logging_config = {"hessian": False, "save": False}
        self.update(eval_logging_config)

    def clear(self) -> None:
        """
        Clear everything in AnaLog.
        """
        self.state.clear()
        self.logging_handler.clear(clear_modules=True)
        self.storage_handler.clear()
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
