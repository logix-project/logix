from functools import partial
from typing import Optional, Callable, List, Dict, Tuple

import torch
import torch.nn as nn
from einops import einsum, reduce

from analog.constants import FORWARD, BACKWARD, GRAD, LOG_TYPES
from analog.storage import StorageHandlerBase
from analog.hessian import HessianHandlerBase
from analog.utils import nested_dict


def compute_per_sample_gradient(fwd, bwd, module):
    """
    Computes the per-sample gradient of a module.

    Args:
        fwd: The forward activations of the module.
        bwd: The backward activations of the module.
        module: The module whose per-sample gradient needs to be computed.
    """
    if isinstance(module, nn.Linear):
        # For linear layers, we can simply compute the outer product of the
        # forward and backward activations.
        outer_product = einsum(bwd, fwd, "... i, ... j -> ... i j")
        return reduce(outer_product, "n ... i j -> n i j", "sum")
    elif isinstance(module, nn.Conv2d):
        # For convolutional layers, we need to unfold the forward activations
        # and compute the outer product of the backward and unfolded forward
        # activations.
        bsz = fwd.shape[0]
        fwd_unfold = torch.nn.functional.unfold(
            fwd,
            module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
        fwd_unfold = fwd_unfold.reshape(bsz, fwd_unfold.shape[1], -1)
        bwd = bwd.reshape(bsz, -1, fwd_unfold.shape[-1])
        grad = torch.einsum("ijk,ilk->ijl", bwd, fwd_unfold)
        shape = [bsz, module.weight.shape[0], -1]
        return grad.reshape(shape)
    elif isinstance(module, nn.Conv1d):
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")


class LoggingHandler:
    def __init__(
        self,
        config: Dict,
        storage_handler: StorageHandlerBase,
        hessian_handler: HessianHandlerBase,
    ) -> None:
        """
        Initializes the LoggingHandler with empty lists for hooks.
        """
        self.config = config

        self.storage_handler = storage_handler
        self.hessian_handler = hessian_handler
        self.hessian_type = hessian_handler.config.get("type", "kfac")

        # Internal states
        self.log = None
        self.hessian = False
        self.save = False
        self.test = False
        self.current_log = None

        # hook handles
        self.modules_to_hook = []
        self.modules_to_name = {}
        self.forward_hooks = []
        self.backward_hooks = []
        self.grad_hooks = []
        self.tensor_hooks = []

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

        activations = inputs[0]
        # In case `mask is not None`, apply the mask to activations
        if self.mask is not None:
            if len(self.mask.shape) != len(activations.shape):
                assert len(self.mask.shape) == len(activations.shape) - 1
                if self.mask.shape[-1] == activations.shape[-2]:
                    activations = activations * self.mask.unsqueeze(-1)
            else:
                if self.mask.shape[-1] == activations.shape[-1]:
                    activations = activations * self.mask

        # If KFAC is used, update the forward covariance
        if self.hessian and self.hessian_type == "kfac":
            self.hessian_handler.update_hessian(
                module, module_name, FORWARD, activations, self.mask
            )

        if FORWARD in self.log:
            if FORWARD not in self.current_log[module_name]:
                self.current_log[module_name][FORWARD] = activations
            else:
                self.current_log[module_name][FORWARD] += activations
            if self.save:
                self.storage_handler.add(module_name, FORWARD, activations)

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

        # If KFAC is used, update the backward covariance
        if self.hessian and self.hessian_type == "kfac":
            self.hessian_handler.update_hessian(
                module, module_name, BACKWARD, grad_outputs[0], self.mask
            )

        if BACKWARD in self.log:
            if BACKWARD not in self.current_log[module_name]:
                self.current_log[module_name][BACKWARD] = grad_outputs[0]
            else:
                self.current_log[module_name][BACKWARD] += grad_outputs[0]
            if self.save:
                self.storage_handler.add(module_name, BACKWARD, grad_outputs[0])

        # As we are only interested in the per-sample gradient, we can delete the
        # gradient of the weight and bias parameters
        del module.weight.grad
        if hasattr(module, "bias") and module.bias is not None:
            del module.bias.grad

    def _grad_hook_fn(
        self,
        module: nn.Module,
        inputs: Tuple[torch.Tensor],
        outputs: Tuple[torch.Tensor],
        module_name: str,
    ) -> None:
        """
        Internal gradient hook function.

        Args:
            module: The module triggering the hook.
            inputs: The input to the module.
            outputs: The output from the module.
            module_name (str): The name of the module.
        """
        assert len(inputs) == 1

        # In case, the same module is used multiple times in the forward pass,
        # we need to accumulate the gradients. We achieve this by using the
        # additional tensor hook on the output of the module.
        def _grad_backward_hook_fn(grad: torch.Tensor):
            if GRAD in self.log:
                per_sample_gradient = compute_per_sample_gradient(
                    inputs[0], grad, module
                )
                if module_name not in self.current_log:
                    self.current_log[module_name] = per_sample_gradient
                else:
                    self.current_log[module_name] += per_sample_gradient
                if self.save:
                    self.storage_handler.add(module_name, GRAD, per_sample_gradient)

        tensor_hook = outputs.register_hook(_grad_backward_hook_fn)
        self.tensor_hooks.append(tensor_hook)

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

    def register_all_module_hooks(self) -> None:
        """
        Register all module hooks.
        """
        for module in self.modules_to_hook:
            # As each hook has its own function, we need to register all hooks
            # separately. We use partial functions to pass the module name to
            # the hook functions.
            module_name = self.get_module_name(module)
            forward_hook = module.register_forward_pre_hook(
                partial(self._forward_hook_fn, module_name=module_name)
            )
            backward_hook = module.register_full_backward_hook(
                partial(self._backward_hook_fn, module_name=module_name)
            )
            grad_hook = module.register_forward_hook(
                partial(self._grad_hook_fn, module_name=module_name)
            )
            self.forward_hooks.append(forward_hook)
            self.backward_hooks.append(backward_hook)
            self.grad_hooks.append(grad_hook)

    def register_all_tensor_hooks(self, tensor_dict: Dict[str, torch.Tensor]) -> None:
        """
        Register all tensor hooks.

        Args:
            tensor_dict (dict): Dictionary containing tensor names as keys and tensors as values.
        """
        for tensor_name, tensor in tensor_dict.items():
            self._tensor_forward_hook_fn(tensor, tensor_name)
            tensor_hook = self.tensor.register_hook(
                partial(self._tensor_backward_hook_fn, tensor_name=tensor_name)
            )
            self.tensor_hooks.append(tensor_hook)

    def set_states(
        self,
        log: List[str],
        hessian: bool,
        save: bool,
        test: bool,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Set the internal states of the logging handler.

        Args:
            log (list): The list of logging types to be used.
            hessian (bool): Whether to compute the hessian.
            save (bool): Whether to save the logging data.
            test (bool): Whether to run in test mode.
        """
        self.log = log
        self.hessian = hessian
        self.save = save
        self.test = test
        self.mask = mask

        self.current_log = nested_dict()

    def add_module(self, module_name: str, module: nn.Module) -> None:
        """
        Add a module to be hooked.

        Args:
            module_name (str): The name of the module.
            module: The module to be hooked.
        """
        self.modules_to_hook.append(module)
        self.modules_to_name[module] = module_name

    def get_module_name(self, module: nn.Module):
        """
        Retrieves the name of a module.

        Args:
            module: The module whose name needs to be retrieved.

        Returns:
            str: The name of the module.
        """
        return self.modules_to_name[module]

    def clear(self, clear_modules: bool = False) -> None:
        """
        Clear all hooks and internal states.

        Args:
            clear_modules (bool): Whether to clear the modules to be hooked.
        """
        self.clear_hooks()
        self.clear_internal_states()
        if clear_modules:
            self.modules_to_hook = []
            self.modules_to_name = {}

    def clear_hooks(self) -> None:
        """
        Clear all registered hooks.
        """
        for hook in self.forward_hooks:
            hook.remove()
        for hook in self.backward_hooks:
            hook.remove()
        for hook in self.grad_hooks:
            hook.remove()
        for hook in self.tensor_hooks:
            hook.remove()

    def clear_internal_states(self) -> None:
        """
        Clear all internal states.
        """
        self.log = None
        self.hessian = False
        self.save = False
        self.test = False
