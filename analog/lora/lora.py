from typing import List, Dict, Any

import torch.nn as nn

from analog.constants import FORWARD, BACKWARD
from analog.state import StatisticState
from analog.lora.modules import LoraLinear, LoraConv2d, LoraEmbedding
from analog.lora.utils import (
    find_parameter_sharing_group,
    _get_submodules,
    find_rank_pca_compression,
    find_rank_pca_covariance,
    pca_rank_by_weight_shape,
)
from analog.lora.utils import find_parameter_sharing_group, _get_submodules
from analog.utils import get_logger, module_check


class LoRAHandler:
    """
    Transforms a model into a Lora model.
    """

    _SUPPORTED_MODULES = {nn.Linear, nn.Conv1d, nn.Conv2d}

    def __init__(self, config: Dict[str, Any], state: StatisticState):
        self.config = config
        self._state = state

        self.parse_config()

    def parse_config(self):
        self.init_strategy = self.config.get("init", "random")
        self.rank_default = self.config.get("rank", 64)
        self.compression_ratio_by_covariance = self.config.get(
            "compression_ratio_by_covariance", None
        )
        self.compression_ratio_by_memory = self.config.get(
            "compression_ratio_by_memory", None
        )
        self.parameter_sharing = self.config.get("parameter_sharing", False)
        self.parameter_sharing_groups = self.config.get(
            "parameter_sharing_groups", None
        )
        self._sanity_check()

    def add_lora(
        self,
        model: nn.Module,
        type_filter: List[nn.Module],
        name_filter: List[str],
        lora_state: Dict[str, Any] = None,
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

            rank_forward = rank_backward = self.rank_default  # default rank

            if lora_state is not None:  # add lora matching the rank of the lora_state
                rank_forward, rank_backward = pca_rank_by_weight_shape(
                    lora_state[name + ".analog_lora_B.weight"].shape, module
                )
            elif self.compression_ratio_by_covariance is not None:
                rank_forward = find_rank_pca_covariance(
                    covariance_state[name][FORWARD],
                    self.compression_ratio_by_covariance,
                )
                rank_backward = find_rank_pca_covariance(
                    covariance_state[name][BACKWARD],
                    self.compression_ratio_by_covariance,
                )
                get_logger().info(
                    f"using adaptive rank_forward = {rank_forward}, rank_backward = {rank_backward} for {name}\n"
                )
            elif self.compression_ratio_by_memory is not None:
                rank_forward = rank_backward = find_rank_pca_compression(
                    module,
                    self.compression_ratio_by_memory,
                )
                get_logger().info(
                    f"using adaptive rank_forward = {rank_forward}, rank_backward = {rank_backward} for {name}\n"
                )

            if self.parameter_sharing and psg not in shared_modules:
                if isinstance(module, nn.Linear):
                    shared_module = nn.Linear(rank_forward, rank_backward, bias=False)
                elif isinstance(module, nn.Conv1d):
                    shared_module = nn.Conv1d(
                        rank_forward, rank_backward, kernel_size=1, bias=False
                    )
                elif isinstance(module, nn.Conv2d):
                    shared_module = nn.Conv2d(
                        rank_forward, rank_backward, kernel_size=1, bias=False
                    )
                shared_modules[psg] = shared_module

            lora_module = lora_cls(
                rank_forward, rank_backward, module, shared_modules.get(psg, None)
            )
            if self.init_strategy == "pca":
                lora_module.pca_init_weight(covariance_state[name])
            lora_module.to(device)

            parent, target, target_name = _get_submodules(model, name)
            setattr(parent, target_name, lora_module)

    def _sanity_check(self):
        if (
            self.init_strategy == "pca"
            and self.compression_ratio_by_covariance is not None
            and self.compression_ratio_by_memory is not None
        ):
            get_logger().warning(
                "compression_ratio_by_covariance and compression_ratio_by_memory are both set. "
                + "compression_ratio_by_covariance will be used."
            )
