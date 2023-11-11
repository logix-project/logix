import torch.nn as nn
from analog.lora.modules import LoraLinear

class LoraTransformer:
    """
    Transforms a model into a Lora model.
    """

    def __init__(self, config):
        self.config = config
        self.parse_config()
        
    def parse_config(self):
        self.type = self.config.get("type", "random")
        self.rank = self.config.get("rank", 64)

    def add_lora_linear(self, model) -> None:
        """
        replace all linear layers with LoraLinear
        """
        device = next(model.parameters()).device
        modules_to_replace = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                modules_to_replace.append((name, module))

        for name, module in modules_to_replace:
            new_module = LoraLinear(self.rank, module).to(device)
            setattr(model, name, new_module)    