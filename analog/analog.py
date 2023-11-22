from typing import Optional, Iterable, Dict, Any, List

import torch
import torch.nn as nn

from analog.config import Config
from analog.constants import FORWARD, BACKWARD, GRAD, LOG_TYPES
from analog.logging import LoggingHandler
from analog.storage import DefaultStorageHandler, MongoDBStorageHandler
from analog.hessian import RawHessianHandler, KFACHessianHandler
from analog.analysis import AnalysisBase
from analog.lora import LoRAHandler
from analog.utils import get_logger


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
            config (str): The path to the YAML configuration file.
        """
        self.project = project

        # Config
        config = Config(config)
        self.config = config

        # Initialize storage, hessian, and logging handlers from config as well as
        # inject dependencies between handlers.
        self.storage_handler = self.build_storage_handler()
        self.hessian_handler = self.build_hessian_handler()
        self.logging_handler = self.build_logging_handler()
        self.lora_handler = self.build_lora_handler()

        # Analysis plugins
        self.analysis_plugins = {}

        # Internal states
        self.log = None
        self.hessian = False
        self.save = False
        self.test = False

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
                isinstance(module, module_type) for module_type in type_filter
            ):
                continue
            if name_filter is not None and not any(
                keyword in name for keyword in name_filter
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
        model: nn.Module,
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
        log: Iterable[str] = [FORWARD, BACKWARD],
        hessian: bool = True,
        save: bool = False,
        test: bool = False,
        strategy: Optional[str] = None,
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
        if strategy is None:
            self.data_id = data_id
            self.log = log
            self.hessian = hessian if not test else False
            self.save = save if not test else False
            self.test = test
        else:
            self.parse_strategy(strategy)

        self.sanity_check(self.data_id, self.log, self.test)

        return self

    def __enter__(self):
        """
        Sets up the context manager.

        This method is automatically called when the `with` statement is used with an `AnaLog` object.
        It sets up the logging environment based on the provided parameters.
        """

        self.storage_handler.set_data_id(self.data_id)

        self.logging_handler.set_states(self.log, self.hessian, self.save, self.test)
        self.logging_handler.register_all_module_hooks()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Clears the internal states and removes all the hooks.

        This method is essential for ensuring that there are no lingering hooks that could
        interfere with further operations on the model or with future logging sessions.
        """
        self.hessian_handler.on_exit(self.logging_handler.current_log)
        self.storage_handler.flush()
        self.logging_handler.clear()

        self.reset()

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
        elif storage_type == "mongodb":
            return MongoDBStorageHandler(storage_config)
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

        if clear:
            self.hessian_handler.clear()
            self.storage_handler.clear()

    def parse_strategy(self, strategy: str) -> None:
        """
        Parses the strategy string to set the internal states.

        Args:
            strategy (str): The strategy string.
        """
        strategy = strategy.lower()
        if strategy == "train":
            self.log = [FORWARD, BACKWARD]
            self.hessian = True
            self.save = False
            self.test = False
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def sanity_check(
        self, data_id: Iterable[Any], log: Iterable[str], test: bool
    ) -> None:
        """
        Performs a sanity check on the provided parameters.
        """
        if len(log) > 0 and len(set(log) - LOG_TYPES) > 0:
            raise ValueError("Invalid value for 'log'.")
        if not test and data_id is None:
            raise ValueError("Must provide data_id for logging.")
        if GRAD in log and len(log) > 1:
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

    def reset(self) -> None:
        """
        Reset the internal states.
        """
        self.log = None
        self.hessian = False
        self.save = False
        self.test = False

    def clear(self) -> None:
        """
        Clear everything in AnaLog.
        """
        self.reset()
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
        get_logger().info(f"Total number of parameters: {repr_dim}\n")
