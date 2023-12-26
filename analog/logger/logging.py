from functools import partial
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn

from analog.batch_info import BatchInfo
from analog.statistic import StatisticState
from analog.logger.config import LoggingConfig
from analog.logger.utils import compute_per_sample_gradient


class HookLogger:
    def __init__(
        self,
        state: StatisticState,
        binfo: BatchInfo,
    ) -> None:
        """
        Initializes the LoggingHandler with empty lists for hooks.
        """
        self.state = state
        self.binfo = binfo
        self.config = LoggingConfig()

        # hooks
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
        log = self.binfo.log[module_name]

        # If `mask` is not None, apply the mask to activations. This is
        # useful for example when you work with sequence models that use
        # padding. In this case, you can use the mask to ignore the padding
        if self.binfo.mask is not None:
            mask = self.binfo.mask
            if len(mask.shape) != len(activations.shape):
                assert len(mask.shape) == len(activations.shape) - 1
                if mask.shape[-1] == activations.shape[-2]:
                    activations = activations * mask.unsqueeze(-1)
            else:
                if mask.shape[-1] == activations.shape[-1]:
                    activations = activations * mask

        if self.config.log["forward"]:
            if "forward" not in log:
                log["forward"] = activations
            else:
                log["forward"] += activations

        for stat_plugin in self.config.statistic["forward"]:
            stat_plugin.update(self.state, self.binfo, module, module_name, "forward", activations)

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

        log = self.binfo.log[module_name]

        if self.config.log["backward"]:
            if "backward" not in log:
                log["backward"] = grad_outputs[0]
            else:
                log["backward"] += grad_outputs[0]

        for stat_plugin in self.config.statistic["backward"]:
            stat_plugin.update(self.state, self.binfo, module, module_name, "backward", grad_outputs[0])

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

        log = self.binfo.log[module_name]

        # In case, the same module is used multiple times in the forward pass,
        # we need to accumulate the gradients. We achieve this by using the
        # additional tensor hook on the output of the module.
        def _grad_backward_hook_fn(grad: torch.Tensor):
            if self.config.log["grad"]:
                per_sample_gradient = compute_per_sample_gradient(
                    inputs[0], grad, module
                )
                if "grad" not in log[module_name]:
                    log["grad"] = per_sample_gradient
                else:
                    log["grad"] += per_sample_gradient

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
        log_state = self.binfo.log[tensor_name]
        log_state[tensor_name]["forward"] = tensor

        for stat_plugin in self.config.statistic["forward"]:
            stat_plugin.update(self.state, self.binfo, None, tensor_name, "forward", tensor)

    def _tensor_backward_hook_fn(self, grad: torch.Tensor, tensor_name: str) -> None:
        """
        Internal backward hook function specifically designed for tensors.

        This method allows you to track the gradients associated with specific
        tensors that are not necessarily tied to any specific module, but are still of interest.

        Args:
            grad: The gradient tensor triggering the hook.
            tensor_name (str): A string identifier for the tensor whose gradient is being tracked.
        """
        log_state = self.binfo.log_state
        log_state[tensor_name]["backward"] = grad

        for stat_plugin in self.config.statistic["backward"]:
            stat_plugin.update(self.state, self.binfo, None, tensor_name, "backward", grad)

    def register_all_module_hooks(self) -> None:
        """
        Register all module hooks.
        """
        for module in self.modules_to_hook:
            # As each hook has its own function, we need to register all hooks
            # separately. We use partial functions to pass the module name to
            # the hook functions.
            module_name = self.modules_to_name[module]
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
            tensor_hook = tensor.register_hook(
                partial(self._tensor_backward_hook_fn, tensor_name=tensor_name)
            )
            self.tensor_hooks.append(tensor_hook)

    def on_exit(self):
        for stat_plugin in self.config.statistic["grad"]:
            for module_name, _ in self.binfo.log.items():
                stat_plugin.update(self.state, self.binfo, None, module_name, "grad")
        self.clear(module=False)

    def add_module(self, module_name: str, module: nn.Module) -> None:
        """
        Add a module to be hooked.

        Args:
            module_name (str): The name of the module.
            module: The module to be hooked.
        """
        self.modules_to_hook.append(module)
        self.modules_to_name[module] = module_name

    def clear(self, hook: bool = True, module: bool = True) -> None:
        """
        Clear all hooks and internal states.

        Args:
            hook (bool): Whether to clear the hooks.
            module (bool): Whether to clear the modules.
        """
        if hook:
            for hook in self.forward_hooks:
                hook.remove()
            for hook in self.backward_hooks:
                hook.remove()
            for hook in self.grad_hooks:
                hook.remove()
            for hook in self.tensor_hooks:
                hook.remove()

        if module:
            self.modules_to_hook = []
            self.modules_to_name = {}
