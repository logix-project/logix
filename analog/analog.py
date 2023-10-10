from functools import partial

from analog.storage import StorageHandler
from analog.covariance import CovarianceHandler


class AnaLog:
    def __init__(
        self, project, config=None, covariance_handler=None, storage_handler=None
    ):
        self.project = project
        self.config = config

        # hook
        self.modules_to_hook = []
        self.modules_to_name = {}
        self.forward_hooks = []
        self.backward_hooks = []
        self.tensor_hooks = []

        # handler
        self.storage_handler = StorageHandler(config)
        self.covariance_handler = covariance_handler or CovarianceHandler(config)

        # log
        self.log = None
        self.hessian = None
        self.save = None
        self.current_log = {}

        # TODO: analysis
        self.data_analyzer = None
        self.weight_analyzer = None

    def _forward_hook_fn(self, module, inputs, module_name):
        """forward hook for module"""
        if self.hessian is not None:
            covariance = self.covariance_handler.compute_covariance(module, inputs)
            self.storage_handler.update_covariance(module_name, "forward", covariance)

        if self.save and "activations" in self.log:
            self.storage_handler.add(module_name, "forward", inputs)

    def _backward_hook_fn(self, module, grad_inputs, grad_outputs, module_name):
        """backward hook for module"""
        del grad_inputs

        if self.hessian is not None:
            covariance = self.covariance_handler.compute_covariance(
                module, grad_outputs
            )
            self.storage_handler.update_covariance(module_name, "backward", covariance)

        if self.save and self.log == "full_activations":
            self.storage_handler.add(
                module_name, "backward", grad_outputs
            )
        elif self.save and self.log == "gradient":
            raise NotImplementedError

    def _tensor_forward_hook_fn(self, tensor, tensor_name):
        """Create forward tensor hook. It's not a real hook, but the name is kept for consistency."""
        if self.hessian is not None:
            covariance = self.covariance_handler.compute_covariance(None, tensor)
            self.storage_handler.update_covariance(tensor_name, "forward", covariance)

        if self.save and "activations" in self.log:
            self.storage_handler.add(
                tensor_name, "forward", tensor
            )

    def _tensor_backward_hook_fn(self, grad, tensor_name):
        """Create backward tensor hook"""
        if self.hessian is not None:
            covariance = self.covariance_handler.compute_covariance(None, grad)
            self.storage_handler.update_covariance(tensor_name, "backward", covariance)

        if self.save and self.log == "full_activations":
            self.storage_handler.add(
                tensor_name, "backward", grad
            )

    def watch(self, model, type_filter=None, name_filter=None):
        """Sets up modules in model to be watched."""
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
        """Sets up the tensor to be watched."""
        for name, tensor in tensor_dict.items():
            self._tensor_forward_hook_fn(tensor, name)
            self.tensor_hooks.append(
                tensor.register_hook(
                    partial(self._tensor_backward_hook_fn, tensor_name=name)
                )
            )

    def get_log(self):
        return self.current_log

    def get_module_name(self, module):
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
        """Sets up the context manager."""
        self.sanity_check(data_id, log, hessian, save)

        self.current_log = None
        self.current_data_id = data_id

        self.log = log
        self.hessian = hessian
        self.save = save
        self.test = test

        self.storage_handler.set_data_id(data_id)

        # register hooks
        for module in self.modules_to_hook:
            module_name = self.get_module_name(module)
            fwd_hook = module.register_forward_hook(
                partial(self._forward_hook_fn, module_name=module_name)
            )
            bwd_hook = module.register_full_backward_hook(
                partial(self._backward_hook_fn, module_name=module_name)
            )
            self.forward_hooks.append(fwd_hook)
            self.backward_hooks.append(bwd_hook)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleans up the context manager."""
        self.clear_for_exit()

    def clear_for_exit(self):
        for hook in self.forward_hooks:
            hook.remove()
        for hook in self.backward_hooks:
            hook.remove()
        for hook in self.tensor_hooks:
            hook.remove()

        self.log = None
        self.hessian = None
        self.save = None
        self.test = None

    def clear(self):
        self.clear_for_exit()
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
