from typing import Optional, Iterable, Dict, Tuple, Any
from functools import partial

import torch
import torch.nn as nn

from analog.hook import HookManager
from analog.config import Config
from analog.constants import FORWARD, BACKWARD, GRAD, LOG_TYPES
from analog.storage import init_storage_handler_from_config
from analog.hessian import init_hessian_handler_from_config
from analog.utils import nested_dict


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
        self.config = Config(config)

        # Hook manager
        self.hook_manager = HookManager()

        # Handlers for storage and Hessian
        self.storage_handler = init_storage_handler_from_config(self.config)
        self.hessian_handler = init_hessian_handler_from_config(self.config)

        # Internal states
        self.modules_to_hook = []
        self.modules_to_name = {}
        self.log = None
        self.hessian = False
        self.save = False

        # Current log
        self.current_log = nested_dict()
        self.data_id = None

    def _forward_hook_fn(
        self, module: nn.Module, inputs: Tuple[torch.Tensor], module_name: str
    ) -> None:
        """
        Internal forward hook function.

        Args:
            module: The module triggering the hook.
            inputs: The input to the module.
            module_name (str): The name of the module.
        """
        assert len(inputs) == 1

        if self.hessian:
            self.hessian_handler.update_hessian(module, module_name, FORWARD, inputs[0])

        if self.save and FORWARD in self.log:
            self.storage_handler.add(module_name, FORWARD, inputs[0])

        self.current_log[module_name][FORWARD] = inputs[0]

    def _backward_hook_fn(
        self,
        module: nn.Module,
        grad_inputs: Tuple[torch.Tensor],
        grad_outputs: Tuple[torch.Tensor],
        module_name: str,
    ) -> None:
        """
        Internal backward hook function.

        Args:
            module: The module triggering the hook.
            grad_inputs: The gradient of the input to the module.
            grad_outputs: The gradient of the output from the module.
            module_name (str): The name of the module.
        """
        assert len(grad_outputs) == 1

        if self.hessian:
            self.hessian_handler.update_hessian(
                module, module_name, BACKWARD, grad_outputs[0]
            )

        if self.save and BACKWARD in self.log:
            self.storage_handler.add(module_name, BACKWARD, grad_outputs[0])
        if self.save and GRAD in self.log:
            raise NotImplementedError

        self.current_log[module_name][BACKWARD] = grad_outputs[0]

    def _tensor_forward_hook_fn(self, tensor: torch.Tensor, tensor_name: str) -> None:
        """
        Internal forward hook function specifically designed for tensors.

        This method allows you to track the activations of tensors that are not
        necessarily tied to any specific module, but are still of interest.

        Args:
            tensor: The tensor triggering the hook.
            tensor_name (str): A string identifier for the tensor, useful for logging.
        """
        if self.hessian:
            self.hessian_handler.update_hessian(None, tensor_name, FORWARD, tensor)

        if self.save and FORWARD in self.log:
            self.storage_handler.add(tensor_name, FORWARD, tensor)

        self.current_log[tensor_name][FORWARD] = tensor

    def _tensor_backward_hook_fn(self, grad: torch.Tensor, tensor_name: str) -> None:
        """
        Internal backward hook function specifically designed for tensors.

        This method allows you to track the gradients associated with specific
        tensors that are not necessarily tied to any specific module, but are still of interest.

        Args:
            grad: The gradient tensor triggering the hook.
            tensor_name (str): A string identifier for the tensor whose gradient is being tracked.
        """
        if self.hessian:
            self.hessian_handler.update_hessian(None, tensor_name, BACKWARD, grad)

        if self.save and BACKWARD in self.log:
            self.storage_handler.add(tensor_name, BACKWARD, grad)

        self.current_log[tensor_name][BACKWARD] = grad

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
            self.modules_to_hook.append(module)
            self.modules_to_name[module] = name

    def watch_activation(self, tensor_dict: Dict[str, torch.Tensor]) -> None:
        """
        Sets up tensors to be watched.

        Args:
            tensor_dict (dict): Dictionary containing tensor names as keys and tensors as values.
        """
        for tensor_name, tensor in tensor_dict.items():
            self._tensor_forward_hook_fn(tensor, tensor_name)
            self.hook_manager.register_tensor_hook(
                tensor, partial(self._tensor_backward_hook_fn, tensor_name=tensor_name)
            )

    def get_log(self):
        """
        Returns the current log.

        Returns:
            dict: The current log.
        """
        return self.current_log

    def get_module_name(self, module: nn.Module) -> str:
        """
        Retrieves the name of a module.

        Args:
            module: The module whose name needs to be retrieved.

        Returns:
            str: The name of the module.
        """
        return self.modules_to_name[module]

    def finalize(self, clear: bool = False) -> None:
        self.hessian_handler.finalize()
        self.storage_handler.finalize()

        if clear:
            self.hessian_handler.clear()
            self.storage_handler.clear()

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

        self.sanity_check(self.data_id, self.log, test)

        return self

    def __enter__(self):
        """
        Sets up the context manager.

        This method is automatically called when the `with` statement is used with an `AnaLog` object.
        It sets up the logging environment based on the provided parameters.
        """

        self.storage_handler.set_data_id(self.data_id)
        self.current_log = nested_dict()

        # register hooks
        for module in self.modules_to_hook:
            module_name = self.get_module_name(module)
            self.hook_manager.register_forward_hook(
                module, partial(self._forward_hook_fn, module_name=module_name)
            )
            self.hook_manager.register_backward_hook(
                module, partial(self._backward_hook_fn, module_name=module_name)
            )

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Clears the internal states and removes all the hooks.

        This method is essential for ensuring that there are no lingering hooks that could
        interfere with further operations on the model or with future logging sessions.
        """
        self.storage_handler.push()
        self.hook_manager.clear_hooks()

        self.reset()

    def reset(self) -> None:
        """
        Reset the internal states.
        """
        self.log = None
        self.hessian = False
        self.save = False

    def clear(self) -> None:
        """
        Clear everything in AnaLog.
        """
        self.reset()
        self.hook_manager.clear_hooks()
        self.modules_to_hook = []
        self.modules_to_name = {}

        self.storage_handler.clear()
        self.hessian_handler.clear()

    def sanity_check(
        self, data_id: Iterable[Any], log: Iterable[str], test: bool
    ) -> None:
        """
        Performs a sanity check on the provided parameters.
        """
        if len(log) > 0 and len(set(log) - LOG_TYPES) > 0:
            raise ValueError("Invalid value for 'track'.")
        if not test and data_id is None:
            raise ValueError("Must provide data_id for logging.")

    def get_hessian_state(self):
        """
        Returns the Hessian state from the Hessian handler.
        """
        return self.hessian_handler.get_hessian_state()

    def get_storage_buffer(self):
        """
        Returns the storage buffer from the storage handler.
        """
        return self.storage_handler.get_buffer()
