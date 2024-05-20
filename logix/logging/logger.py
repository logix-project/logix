from functools import partial
from typing import Optional, Dict, Tuple, Any

import torch
import torch.nn as nn

from logix.batch_info import BatchInfo
from logix.config import LoggingConfig
from logix.state import LogIXState
from logix.logging.option import LogOption
from logix.logging.log_saver import LogSaver
from logix.logging.utils import compute_per_sample_gradient
from logix.utils import get_logger


class HookLogger:
    def __init__(
        self,
        config: LoggingConfig,
        state: LogIXState,
        binfo: BatchInfo,
    ) -> None:
        """
        Initializes the LoggingHandler with empty lists for hooks.
        """
        self.state = state
        self.binfo = binfo
        self.opt = LogOption()

        # parse config
        self.cpu_offload = config.cpu_offload
        self.dtype = config.get_dtype()

        # log saver
        self.log_saver = LogSaver(config=config, state=self.state)

        # hooks
        self.modules_to_hook = []
        self.modules_to_name = {}
        self.forward_hooks = []
        self.backward_hooks = []
        self.grad_hooks = []
        self.tensor_hooks = []

    def log(self, data_id: Any, mask: Optional[torch.Tensor] = None):
        """
        Add log state on exit.

        Args:
            data_id: The data ID associated with the current batch.
            mask (torch.Tensor): A mask to be applied to the activations.
        """
        log = self.binfo.log.copy()
        self.binfo.clear()
        self.binfo.data_id = data_id
        self.binfo.mask = mask

        self.update()

        return log

    def save_log(self):
        # save log to disk
        if any(self.opt.save.values()):
            self.log_saver.buffer_write(binfo=self.binfo)
            self.log_saver.flush()

    def update(self):
        # Update statistics
        for stat in self.opt.statistic["grad"]:
            for module_name, _ in self.binfo.log.items():
                stat.update(
                    state=self.state,
                    binfo=self.binfo,
                    module=None,
                    module_name=module_name,
                    log_type="grad",
                    data=None,
                    cpu_offload=self.cpu_offload,
                )

        # Wait for all asynchronous CUDA operations to finish
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()

        # Write and flush the buffer if necessary
        self.save_log()

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

        if self.dtype is not None:
            activations = activations.to(dtype=self.dtype)

        if self.opt.log["forward"]:
            if "forward" not in log:
                log["forward"] = activations
            else:
                log["forward"] += activations

        for stat in self.opt.statistic["forward"]:
            stat.update(
                state=self.state,
                binfo=self.binfo,
                module=module,
                module_name=module_name,
                log_type="forward",
                data=activations,
                cpu_offload=self.cpu_offload,
            )

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

        error = grad_outputs[0]
        log = self.binfo.log[module_name]

        if self.dtype is not None:
            error = error.to(dtype=self.dtype)

        if self.opt.log["backward"]:
            if "backward" not in log:
                log["backward"] = error
            else:
                log["backward"] += error

        for stat in self.opt.statistic["backward"]:
            stat.update(
                state=self.state,
                binfo=self.binfo,
                module=module,
                module_name=module_name,
                log_type="backward",
                data=error,
                cpu_offload=self.cpu_offload,
            )

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
            if self.opt.log["grad"]:
                per_sample_gradient = compute_per_sample_gradient(
                    inputs[0], grad, module
                )

                if self.dtype is not None:
                    per_sample_gradient = per_sample_gradient.to(dtype=self.dtype)

                if "grad" not in log:
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
        log = self.binfo.log[tensor_name]

        if self.dtype is not None:
            tensor = tensor.to(dtype=self.dtype)

        log["forward"] = tensor

        for stat in self.opt.statistic["forward"]:
            stat.update(
                state=self.state,
                binfo=self.binfo,
                module=None,
                module_name=tensor_name,
                log_type="forward",
                data=tensor,
                cpu_offload=self.cpu_offload,
            )

    def _tensor_backward_hook_fn(self, grad: torch.Tensor, tensor_name: str) -> None:
        """
        Internal backward hook function specifically designed for tensors.

        This method allows you to track the gradients associated with specific
        tensors that are not necessarily tied to any specific module, but are still of interest.

        Args:
            grad: The gradient tensor triggering the hook.
            tensor_name (str): A string identifier for the tensor whose gradient is being tracked.
        """
        log = self.binfo.log[tensor_name]

        if self.dtype is not None:
            grad = grad.to(dtype=self.dtype)

        log["backward"] = grad

        for stat in self.opt.statistic["backward"]:
            stat.update(
                state=self.state,
                binfo=self.binfo,
                module=None,
                module_name=tensor_name,
                log_type="backward",
                data=grad,
                cpu_offload=self.cpu_offload,
            )

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

    def add_module(self, module_name: str, module: nn.Module) -> None:
        """
        Add a module to be hooked.

        Args:
            module_name (str): The name of the module.
            module: The module to be hooked.
        """
        self.modules_to_hook.append(module)
        self.modules_to_name[module] = module_name

    def setup(self, log_option_kwargs: Dict[str, Any]) -> None:
        """
        Update logging configurations.

        Args:
            log_option_kwargs: Logging configurations.
        """
        self.opt.setup(log_option_kwargs)

    def finalize(self):
        """
        Dump everything in the buffer to a disk.
        """
        self.log_saver.finalize()

    def clear(
        self, hook: bool = True, module: bool = True, buffer: bool = True
    ) -> None:
        """
        Clear all hooks and internal states.

        Args:
            hook (bool): Whether to clear the hooks.
            module (bool): Whether to clear the modules.
        """
        if hook:
            for hook in self.forward_hooks:
                hook.remove()
            self.forward_hooks.clear()
            for hook in self.backward_hooks:
                hook.remove()
            self.backward_hooks.clear()
            for hook in self.grad_hooks:
                hook.remove()
            self.grad_hooks.clear()
            for hook in self.tensor_hooks:
                hook.remove()
            self.tensor_hooks.clear()

        if module:
            self.modules_to_hook = []
            self.modules_to_name = {}

        if buffer:
            self.log_saver.buffer_clear()
