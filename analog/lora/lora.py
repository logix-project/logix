import torch.nn as nn

from analog.lora.modules import LoraLinear
from analog.hessian import HessianHandlerBase


class LoRAHandler:
    """
    Transforms a model into a Lora model.
    """

    _SUPPORTED_MODULES = {nn.Linear, nn.Conv1d, nn.Conv2d}

    def __init__(self, config, hessian_handler: HessianHandlerBase):
        self.hessian_handler = hessian_handler

        self.config = config
        self.parse_config()

    def parse_config(self):
        self.type = self.config.get("type", "random")
        self.rank = self.config.get("rank", 64)

    def add_lora(self, model, type_filter, name_filter):
        """
        Add LoRA modules to a model.
        """
        device = next(model.parameters()).device
        for name, module in model.named_modules():
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

            # Add Lora to filtered modules
            if isinstance(module, nn.Linear):
                new_module = LoraLinear(self.rank, module).to(device)
                setattr(model, name, new_module)
            elif isinstance(module, nn.Conv1d):
                raise NotImplementedError
            elif isinstance(module, nn.Conv2d):
                raise NotImplementedError
