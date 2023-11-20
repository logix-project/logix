from typing import List

import torch.nn as nn

from analog.lora.modules import LoraLinear


def find_parameter_sharing_group(
    module_name: str, parameter_sharing_groups: List[str] = None
):
    if parameter_sharing_groups is None:
        return "analog_lora_none"

    found_groups = [psg for psg in parameter_sharing_groups if psg in module_name]
    assert (
        len(found_groups) == 1
    ), "Each module can belong to only one parameter sharing group."
    return found_groups[0]


class LoRAHandler:
    """
    Transforms a model into a Lora model.
    """

    _SUPPORTED_MODULES = {nn.Linear, nn.Conv1d, nn.Conv2d}

    def __init__(self, config):
        self.config = config
        self.parse_config()

    def parse_config(self):
        self.init_strategy = self.config.get("init", "random")
        self.rank = self.config.get("rank", 64)

    def add_lora(
        self,
        model: nn.Module,
        type_filter: List[nn.Module],
        name_filter: List[str],
        hessian_state=None,
        parameter_sharing=False,
        parameter_sharing_groups=None,
    ):
        """
        Add LoRA modules to a model.
        """
        if self.init_strategy == "pca" and hessian_state is None:
            raise ValueError("hessian_state must be provided for pca LoRA")

        shared_modules = {}
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
            lora_cls = None
            if isinstance(module, nn.Linear):
                lora_cls = LoraLinear
            elif isinstance(module, nn.Conv1d):
                raise NotImplementedError
            elif isinstance(module, nn.Conv2d):
                raise NotImplementedError

            psg = find_parameter_sharing_group(name, parameter_sharing_groups)
            if parameter_sharing and psg not in shared_modules:
                if isinstance(module, nn.Linear):
                    shared_module = nn.Linear(self.rank, self.rank, bias=False)
                elif isinstance(module, nn.Conv1d):
                    shared_module = nn.Conv1d(
                        self.rank, self.rank, kernel_size=1, bias=False
                    )
                elif isinstance(module, nn.Conv2d):
                    shared_module = nn.Conv2d(
                        self.rank, self.rank, kernel_size=1, bias=False
                    )
                shared_modules[psg] = shared_module

            lora_module = lora_cls(self.rank, module, shared_modules.get(psg, None))
            lora_module.init_weight(self.init_strategy, hessian_state[name])
            lora_module.to(device)

            setattr(model, name, lora_module)
