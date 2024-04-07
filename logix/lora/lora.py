from typing import List, Dict, Any

import torch.nn as nn

from logix.state import LogIXState
from logix.lora.modules import LoraLinear, LoraConv2d, LoraEmbedding
from logix.lora.utils import find_parameter_sharing_group, _get_submodules
from logix.utils import get_logger, module_check


class LoRAHandler:
    """
    Transforms a model into a Lora model.
    """

    _SUPPORTED_MODULES = {nn.Linear, nn.Conv1d, nn.Conv2d}

    def __init__(self, config: Dict[str, Any], state: LogIXState):
        self._state = state

        self.init_strategy = config.init
        self.rank = config.rank
        self.parameter_sharing = config.parameter_sharing
        self.parameter_sharing_groups = config.parameter_sharing_groups

    def add_lora(
        self,
        model: nn.Module,
        type_filter: List[nn.Module],
        name_filter: List[str],
    ):
        """
        Add LoRA modules to a model.
        """
        covariance_state = self._state.get_covariance_state()
        if self.init_strategy == "pca" and len(covariance_state) == 0:
            get_logger().warning(
                "Hessian state not provided. Using random initialization instead."
            )
            self.init_strategy = "random"

        shared_modules = {}
        device = next(model.parameters()).device
        for name, module in model.named_modules():
            if not module_check(
                module=module,
                module_name=name,
                supported_modules=self._SUPPORTED_MODULES,
                type_filter=type_filter,
                name_filter=name_filter,
            ):
                continue

            # Add Lora to filtered modules
            lora_cls = None
            if isinstance(module, nn.Linear):
                lora_cls = LoraLinear
            elif isinstance(module, nn.Conv1d):
                raise NotImplementedError
            elif isinstance(module, nn.Conv2d):
                lora_cls = LoraConv2d
            elif isinstance(module, nn.Embedding):
                lora_cls = LoraEmbedding

            psg = find_parameter_sharing_group(name, self.parameter_sharing_groups)
            if self.parameter_sharing and psg not in shared_modules:
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
            if self.init_strategy == "pca":
                lora_module.pca_init_weight(covariance_state[name])
            lora_module.to(device)

            parent, _, target_name = _get_submodules(model, name)
            setattr(parent, target_name, lora_module)
