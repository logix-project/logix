from functools import partial

from analog.storage import StorageHandler
from analog.covariance import CovarianceHandler
from analog.hash import SHA256Hasher


class AnaLog:
    def __init__(self, project, config=None, covariance_handler=None, data_hasher=None):
        self.project = project
        self.config = config

        # hook
        self.modules_to_hook = []
        self.forward_hooks = []
        self.backward_hooks = []
        self.tensor_hooks = []

        # handler
        self.storage_handler = StorageHandler(config)
        self.covariance_handler = covariance_handler or CovarianceHandler(config)
        self.data_hasher = data_hasher or SHA256Hasher(config)

        # log
        self.log = None
        self.hessian = None
        self.save = None
        self.current_log = None

        # TODO: analysis
        self.data_analyzer = None
        self.weight_analyzer = None

    def _hash_tensor(self, tensor):
        """Computes a hash for a tensor using the provided data hasher."""
        return self.data_hasher.hash(tensor)

    def _forward_hook_fn(self, module, inputs, module_name):
        if self.hessian is not None:
            covariance = self.covariance_handler.compute_covariance(module, inputs)
            self.storage_handler.update_covariance(module_name, "forward", covariance)

        # offload
        inputs = inputs.cpu()

        if self.save and "activations" in self.log:
            self.storage_handler.push(self.current_data_hash, module, "forward", inputs)

    def _backward_hook_fn(self, module, grad_inputs, grad_outputs, module_name):
        del grad_inputs

        if self.hessian is not None:
            covariance = self.covariance_handler.compute_covariance(
                module, grad_outputs
            )
            self.storage_handler.update_covariance(module_name, "backward", covariance)

        # offload
        grad_outputs = grad_outputs.cpu()

        if self.save and self.log == "full_activations":
            self.storage_handler.push(
                self.current_data_hash, module, "backward", grad_outputs
            )
        elif self.save and self.log == "gradient":
            raise NotImplementedError

    def _tensor_hook_fn(self, grad, tensor_name):
        if self.hessian is not None:
            covariance = self.covariance_handler.compute_covariance(None, grad)
            self.storage_handler.update_covariance(tensor_name, "backward", covariance)

        # offload
        grad = grad.cpu()

        if self.save and self.log == "full_activations":
            self.storage_handler.push(
                self.current_data_hash, tensor_name, "backward", grad
            )

    def watch(self, model, type_filter=None, name_filter=None):
        """Sets up the model to be watched."""
        self.model = model

        for name, module in self.model.named_modules():
            if type_filter is not None and not any(
                isinstance(module, module_type) for module_type in type_filter
            ):
                continue
            if name_filter is not None and not any(
                keyword in name for keyword in name_filter
            ):
                continue
            self.modules_to_hook.append(module)

    def watch_activation(self, tensor_dict):
        """Sets up the tensor to be watched."""
        for name, tensor in tensor_dict.items():
            self.tensor_hooks.append(
                tensor.register_hook(partial(self._tensor_hook_fn, tensor_name=name))
            )

            if self.hessian is not None:
                covariance = self.covariance_handler.compute_covariance(
                    None, tensor_dict
                )
                self.storage_handler.update_covariance(name, "forward", covariance)

            # offload
            tensor = tensor.cpu()

            if self.save and self.log == "full_activations":
                self.storage_handler.push(
                    self.current_data_hash, name, "forward", tensor
                )

    def get_log(self):
        raise self.current_log

    def finalize(self):
        raise NotImplementedError

    def __enter__(
        self,
        data=None,
        log=None,
        hessian=None,
        save=None,
        test=None,
    ):
        """Sets up the context manager."""
        self.sanity_check(data, log, hessian, save)

        self.current_log = None
        self.current_data_hash = self._hash_tensor(data)

        self.log = log
        self.hessian = hessian
        self.save = save
        self.test = test

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
