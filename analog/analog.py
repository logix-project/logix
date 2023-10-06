from analog.covariance import CovarianceHandler
from analog.hash import SHA256Hasher


class AnaLog:
    def __init__(self, project, config=None, cov_computer=None, data_hasher=None):
        self.project = project
        self.config = config

        self.forward_hooks = []
        self.backward_hooks = []
        self.data_map = {}
        self.forward_covariances = {}
        self.backward_covariances = {}

        default_cov_computer = CovarianceHandler()
        self.cov_computer = cov_computer

        self.data_hasher = data_hasher if data_hasher else SHA256Hasher()

        # Track
        self.track = None
        self.covariance_algo = None
        self.save = None

    def _hash_tensor(self, tensor):
        """Computes a hash for a tensor using the provided data hasher."""
        return self.data_hasher.hash(tensor)

    def _forward_hook_fn(self, module, input, output):
        if self.save_activations:
            self.data_map[self.current_input_hash] = (output, None)

        if self.compute_forward_cov:
            cov_computer = self.cov_computers.get(module)
            cov = cov_computer.compute(output)
            self.forward_covariances[module] = cov

    def _backward_hook_fn(self, module, grad_input, grad_output):
        if self.save_gradients:
            if self.current_input_hash in self.data_map:
                activation, _ = self.data_map[self.current_input_hash]
                self.data_map[self.current_input_hash] = (activation, grad_output[0])

        if self.compute_backward_cov:
            cov_computer = self.cov_computers.get(module)
            cov = cov_computer.compute(grad_output[0])
            self.backward_covariances[module] = cov

    def hook_by_type(self, module_type):
        """Hooks all modules of a specific type."""
        for _, module in self.model.named_modules():
            if isinstance(module, module_type):
                self.modules_to_hook.append(module)

    def hook_by_name(self, module_name):
        """Hooks a module based on its name."""
        for name, module in self.model.named_modules():
            if name == module_name:
                self.modules_to_hook.append(module)
                break

    def __enter__(
        self,
        data=None,
        track=None,
        covariance=None,
        save=None,
    ):
        self.sanity_check(data, track, covariance, save)

        self.current_data_hash = self._hash_tensor(data)

        self.track = track
        self.covariance_algo = covariance
        self.save = save

        for hook in self.forward_hooks:
            hook.remove()
        for hook in self.backward_hooks:
            hook.remove()
        self.forward_hooks.clear()
        self.backward_hooks.clear()

        for module in self.modules_to_hook:
            fwd_hook = module.register_forward_hook(self._forward_hook_fn)
            bwd_hook = module.register_backward_hook(self._backward_hook_fn)
            self.forward_hooks.append(fwd_hook)
            self.backward_hooks.append(bwd_hook)

        return self.data_map, self.forward_covariances, self.backward_covariances

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self.forward_hooks:
            hook.remove()
        for hook in self.backward_hooks:
            hook.remove()
        self.clear()

    def clear(self):
        self.

    def sanity_check(self, data, track, covariance, save):
        if save and data is None:
            raise ValueError("Must provide data to save gradients.")
        if track is not None and track not in {"gradient", "full_activations", "activations"}:
            raise ValueError("Invalid value for 'track'.")
        if covariance is not None and covariance not in {"kfac", "shampoo"}:
            raise ValueError("Invalid value for 'covariance'.")

