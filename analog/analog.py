from typing import Optional, Dict, Any
import torch.nn as nn
import torch.Tensor as Tensor

from analog.database import DatabaseHandlerBase, JsonDatabaseHandler
from analog.covariance import CovarianceHandler


class AnaLog:
    def __init__(
        self,
        project: str,
        config: Optional[Dict[str, Any]] = None,
        covariance_handler: Optional[CovarianceHandler] = None,
        database_handler: Optional[DatabaseHandlerBase] = None,
    ) -> None:
        """
        Initializes the AnaLog class for neural network logging.

        Args:
            project (str): The name or identifier of the project.
            config (dict, optional): Configuration parameters for AnaLog.
            covariance_handler (object, optional): A handler for computing covariance.
            database_handler (object, optional): A handler for storing logs.
        """
        self.project = project
        self.config = config

        # Hook manager
        self.hook_manager = HookManager()

        # Handlers for database and Hessian
        self.database_handler = database_handler or JsonDatabaseHandler(config)
        self.covariance_handler = covariance_handler or CovarianceHandler(config)

        # Internal states
        self.modules_to_hook = []
        self.modules_to_name = {}
        self.log = None
        self.hessian = None
        self.save = None

    def _forward_hook_fn(
        self, module: nn.Module, inputs: Tensor, module_name: str
    ) -> None:
        """
        Internal forward hook function.

        Args:
            module: The module triggering the hook.
            inputs: The input to the module.
            module_name (str): The name of the module.
        """
        if self.hessian is not None:
            covariance = self.covariance_handler.update_covariance(
                module, "forward", inputs
            )

        inputs = inputs.cpu()

        if self.save and "activations" in self.log:
            self.database_handler.add(module_name, "forward", inputs)

    def _backward_hook_fn(
        self,
        module: nn.Module,
        grad_inputs: Optional[Tensor],
        grad_outputs: Optional[Tensor],
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
        del grad_inputs

        if self.hessian is not None:
            covariance = self.covariance_handler.update_covariance(
                module, "backward", grad_outputs
            )

        grad_outputs = grad_outputs.cpu()

        if self.save and self.log == "full_activations":
            self.database_handler.add(module_name, "backward", grad_outputs)
        elif self.save and self.log == "gradient":
            raise NotImplementedError

        self.current_log[module_name]["backward"] = grad_outputs

    def _tensor_forward_hook_fn(self, tensor: Tensor, tensor_name: str) -> None:
        """
        Internal forward hook function specifically designed for tensors.

        This method allows you to track the activations of tensors that are not
        necessarily tied to any specific module, but are still of interest.

        Args:
            tensor: The tensor triggering the hook.
            tensor_name (str): A string identifier for the tensor, useful for logging.
        """
        if self.hessian is not None:
            covariance = self.covariance_handler.update_covariance(
                None, "forward", tensor
            )

        # offload data to cpu
        tensor = tensor.cpu()

        if self.save and "activations" in self.log:
            self.database_handler.add(tensor_name, "forward", tensor)

        self.current_log[module_name]["forward"] = tensor

    def _tensor_backward_hook_fn(self, grad: Tensor, tensor_name: str) -> None:
        """
        Internal backward hook function specifically designed for tensors.

        This method allows you to track the gradients associated with specific
        tensors that are not necessarily tied to any specific module, but are still of interest.

        Args:
            grad: The gradient tensor triggering the hook.
            tensor_name (str): A string identifier for the tensor whose gradient is being tracked.
        """
        if self.hessian is not None:
            covariance = self.covariance_handler.update_covariance(
                None, "backward", grad
            )

        # offload data to cpu
        grad = grad.cpu()

        if self.save and self.log == "full_activations":
            self.database_handler.add(tensor_name, "backward", grad)

        self.current_log[module_name]["backward"] = grad

    def watch(self, model, type_filter=None, name_filter=None):
        """
        Sets up modules in the model to be watched.

        Args:
            model: The neural network model.
            type_filter (list, optional): List of types of modules to be watched.
            name_filter (list, optional): List of keyword names for modules to be watched.
        """
        self.model = model

        for name, module in self.model.named_modules():
            # only consider the leaf module
            if len(list(module.children())) > 0:
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

    def watch_activation(self, tensor_dict):
        """
        Sets up tensors to be watched.

        Args:
            tensor_dict (dict): Dictionary containing tensor names as keys and tensors as values.
        """
        for tensor_name, tensor in tensor_dict.items():
            self._tensor_forward_hook_fn(tensor, name)
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

    def get_module_name(self, module):
        """
        Retrieves the name of a module.

        Args:
            module: The module whose name needs to be retrieved.

        Returns:
            str: The name of the module.
        """
        return self.modules_to_name[module]

    def finalize(self):
        raise NotImplementedError

    def __enter__(
        self,
        data_id=None,
        log=None,
        hessian=None,
        save=None,
        test=None,
    ):
        """
        Sets up the context manager.

        This method is automatically called when the `with` statement is used with an `AnaLog` object.
        It sets up the logging environment based on the provided parameters.

        Args:
            data_id: A unique identifier associated with the data for the logging session.
            log (str, optional): Specifies which data to log (e.g. "gradient", "full_activations").
            hessian (str, optional): Specifies if and which kind of hessian matrix approximation is required (e.g. "kfac").
            save (bool, optional): Whether to save the logs or not.
            test: Reserved for future use.

        Returns:
            self: Returns the instance of the AnaLog object.
        """
        self.sanity_check(data_id, log, hessian, save)

        self.current_log = None

        self.log = log
        self.hessian = hessian
        self.save = save
        self.test = test

        self.database_handler.set_data_id(data_id)

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

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Clears the internal states and removes all the hooks.

        This method is essential for ensuring that there are no lingering hooks that could
        interfere with further operations on the model or with future logging sessions.
        """
        self.database_handler.push()
        self.database_handler.clear()
        self.hook_manager.clear_hooks()

        self.clear()

    def clear(self):
        self.log = None
        self.hessian = None
        self.save = None
        self.test = None

    def clear_all(self):
        self.clear()
        self.modules_to_hook = []
        self.modules_to_name = {}

    def sanity_check(self, data, log, hessian, save):
        if save and data is None:
            raise ValueError("Must provide data to save gradients.")
        if log is not None and log not in {
            "gradient",
            "full_activations",
            "activations",
        }:
            raise ValueError("Invalid value for 'track'.")
        if hessian is not None and hessian not in {"kfac", "shampoo"}:
            raise ValueError("Invalid value for 'covariance'.")
