from typing import List, Dict, Any

import torch.nn as nn

from analog.constants import FORWARD, BACKWARD
from analog.state import AnaLogState
from analog.lora.modules import LoraLinear, LoraConv2d, LoraEmbedding
from analog.lora.utils import (
    find_parameter_sharing_group,
    _get_submodules,
    find_rank_pca_covariance,
)
from analog.utils import get_logger


class LoRAHandler:
    """
    Transforms a model into a Lora model.
    """

    _SUPPORTED_MODULES = {nn.Linear, nn.Conv1d, nn.Conv2d}

    def __init__(self, config: Dict[str, Any], state: AnaLogState):
        self.config = config
        self._state = state

        self.parse_config()

    def parse_config(self):
        self.init_strategy = self.config.get("init", "random")
        self.rank = self.config.get("rank", 64)
        self.adaptive_threshold = self.config.get("adaptive_threshold", None)
        self.parameter_sharing = self.config.get("parameter_sharing", False)
        self.parameter_sharing_groups = self.config.get(
            "parameter_sharing_groups", None
        )

    def add_lora(
        self,
        model: nn.Module,
        type_filter: List[nn.Module],
        name_filter: List[str],
    ):
        """
        Add LoRA modules to a model.
        """
        hessian_state = self._state.get_hessian_state()
        if self.init_strategy == "pca" and len(hessian_state) == 0:
            get_logger().warning(
                "Hessian state not provided. Using random initialization instead."
            )
            self.init_strategy = "random"

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
                lora_cls = LoraConv2d
            elif isinstance(module, nn.Embedding):
                lora_cls = LoraEmbedding

            psg = find_parameter_sharing_group(name, self.parameter_sharing_groups)

            rank = self.rank
            if self.adaptive_threshold is not None:
                rank_forward = find_rank_pca_covariance(
                    hessian_state[name][FORWARD], self.adaptive_threshold
                )
                rank_backward = find_rank_pca_covariance(
                    hessian_state[name][BACKWARD], self.adaptive_threshold
                )
                rank = max(rank_forward, rank_backward)
                get_logger().info(f"using adaptive r = {rank} for {name}\n")

            if self.parameter_sharing and psg not in shared_modules:
                if isinstance(module, nn.Linear):
                    shared_module = nn.Linear(rank, rank, bias=False)
                elif isinstance(module, nn.Conv1d):
                    shared_module = nn.Conv1d(rank, rank, kernel_size=1, bias=False)
                elif isinstance(module, nn.Conv2d):
                    shared_module = nn.Conv2d(rank, rank, kernel_size=1, bias=False)
                shared_modules[psg] = shared_module

            lora_module = lora_cls(rank, module, shared_modules.get(psg, None))
            if self.init_strategy == "pca":
                lora_module.pca_init_weight(self.init_strategy, hessian_state[name])
            lora_module.to(device)

            parent, target, target_name = _get_submodules(model, name)
            setattr(parent, target_name, lora_module)
