class HookManager:
    def __init__(self):
        """
        Initializes the HookManager with empty lists for hooks.
        """
        self.forward_hooks = []
        self.backward_hooks = []
        self.tensor_hooks = []

    def register_forward_hook(self, module, hook_fn):
        """
        Register a forward hook on the given module.

        Args:
            module: The module to register the hook on.
            hook_fn: The function to be called when the hook is triggered.

        Returns:
            The hook handle.
        """
        hook = module.register_forward_pre_hook(hook_fn)
        self.forward_hooks.append(hook)
        return hook

    def register_backward_hook(self, module, hook_fn):
        """
        Register a backward hook on the given module.

        Args:
            module: The module to register the hook on.
            hook_fn: The function to be called when the hook is triggered.

        Returns:
            The hook handle.
        """
        hook = module.register_full_backward_hook(hook_fn)
        self.backward_hooks.append(hook)
        return hook

    def register_tensor_hook(self, tensor, hook_fn):
        """
        Register a hook on the given tensor.

        Args:
            tensor: The tensor to register the hook on.
            hook_fn: The function to be called when the hook is triggered.

        Returns:
            The hook handle.
        """
        hook = tensor.register_hook(hook_fn)
        self.tensor_hooks.append(hook)
        return hook

    def clear_hooks(self):
        """
        Clear all registered hooks.
        """
        for hook in self.forward_hooks:
            hook.remove()
        for hook in self.backward_hooks:
            hook.remove()
        for hook in self.tensor_hooks:
            hook.remove()
