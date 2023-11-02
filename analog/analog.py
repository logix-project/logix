from typing import Optional, Iterable, Dict, Tuple, Any

import torch
import torch.nn as nn

from analog.config import Config
from analog.constants import FORWARD, BACKWARD, GRAD, LOG_TYPES
from analog.logging import LoggingHandler
from analog.storage import init_storage_handler_from_config
from analog.hessian import init_hessian_handler_from_config
from analog.analysis import AnalysisBase


class AnaLog:
    _SUPPORTED_MODULES = {nn.Linear, nn.Conv2d}

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
        self.config = config.get_global_config()

        # Initialize storage, hessian, and logging handlers from config as well as
        # inject dependencies between handlers.
        self.storage_handler = init_storage_handler_from_config(
            config.get_storage_config()
        )
        self.hessian_handler = init_hessian_handler_from_config(
            config.get_hessian_config()
        )
        self.logging_handler = LoggingHandler(
            config.get_logging_config(), self.storage_handler, self.hessian_handler
        )

        # Analysis plugins
        self.analysis_plugins = {}

        # Internal states
        self.log = None
        self.hessian = False
        self.save = False
        self.test = False

    def watch(self, model, type_filter=None, name_filter=None) -> None:
        """
        Sets up modules in the model to be watched.

        Args:
            model: The neural network model.
            type_filter (list, optional): List of types of modules to be watched.
            name_filter (list, optional): List of keyword names for modules to be watched.
        """
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
            self.logging_handler.add_module(name, module)

    def watch_activation(self, tensor_dict: Dict[str, torch.Tensor]) -> None:
        """
        Sets up tensors to be watched.

        Args:
            tensor_dict (dict): Dictionary containing tensor names as keys and tensors as values.
        """
        self.logging_handler.register_all_tensor_hooks(tensor_dict)

    def add_analysis(self, analysis_dict: Dict[str, AnalysisBase]) -> None:
        """
        Adds analysis plugins to AnaLog.

        Args:
            analysis_dict (dict): Dictionary containing analysis names as keys and analysis classes as values.
        """
        for analysis_name, analysis_cls in analysis_dict.items():
            if hasattr(self, analysis_name):
                raise ValueError(f"Analysis name {analysis_name} is reserved.")
            setattr(self, analysis_name, analysis_cls(self))
            self.analysis_plugins[analysis_name] = getattr(self, analysis_name)

    def remove_analysis(self, analysis_name: str) -> None:
        """
        Removes analysis plugins from AnaLog.

        Args:
            analysis_name (str): Name of the analysis to be removed.
        """
        if analysis_name not in self.analysis_plugins:
            print(f"Analysis {analysis_name} does not exist. Nothing to remove.")
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
        self.data_id = data_id
        self.log = log
        self.hessian = hessian if not test else False
        self.save = save if not test else False
        self.test = test

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
        self.storage_handler.flush()
        self.logging_handler.clear()

        self.reset()

    def build_log_data_loader(self):
        """
        Constructs the log dataset from the storage handler.
        """
        raise NotImplementedError

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

    def get_hessian_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns the Hessian state from the Hessian handler.
        """
        return self.hessian_handler.get_hessian_state()

    def hessian_inverse(self, override: bool = False) -> None:
        """
        Compute the inverse of the Hessian.
        """
        self.hessian_handler.hessian_inverse(override)

    def finalize(self, clear: bool = False) -> None:
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
        self.logging_handler.clear()
        self.storage_handler.clear()
        self.hessian_handler.clear()
        for key in self.analysis_plugins:
            self.remove_analysis(key)
