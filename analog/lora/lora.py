import torch.nn as nn
from analog.lora.modules import LoraLinear

class LoRAHandler:
    """
    Transforms a model into a Lora model.
    """

    def __init__(self, config):
        self.config = config
        self.parse_config()
        
    def parse_config(self):
        self.type = self.config.get("type", "random")
        self.rank = self.config.get("rank", 64)

    def add_lora(self, model, type_filter, name_filter):
        """
        Add LoRA modules to a model.
        """
        if type_filter is None or nn.Linear in type_filter:
            self.add_lora_linear(model, name_filter)

        if type_filter is None or nn.Conv2d in type_filter:
            self.add_lora_conv2d(model, name_filter)

    def add_lora_linear(self, model, name_filter) -> None:
        """
        replace linear layers that match the name_filter with LoraLinear
        """
        device = next(model.parameters()).device
        modules_to_replace = []

        for name, module in model.named_modules():
            name_not_filtered = name_filter is None or any(keyword in name for keyword in name_filter)
            if isinstance(module, nn.Linear) and name_not_filtered:
                modules_to_replace.append((name, module))

        for name, module in modules_to_replace:
            new_module = LoraLinear(self.rank, module).to(device)
            setattr(model, name, new_module)

    def add_lora_conv2d(self, name, name_filter) -> None:
        """
        replace all conv layers that match the name_filter with LoraConv
        """
        pass