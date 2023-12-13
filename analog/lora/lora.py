from typing import Any, Dict, List

import torch.nn as nn
from analog.constants import BACKWARD, FORWARD
from analog.lora.modules import LoraConv2d, LoraLinear
from analog.lora.utils import _get_submodules, find_parameter_sharing_group
from analog.state import AnaLogState
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
        self.parameter_sharing = self.config.get("parameter_sharing", False)
        self.parameter_sharing_groups = self.config.get(
            "parameter_sharing_groups", None
        )

    def _named_modules(
        self,
        model: nn.Module,
        type_filter: List[nn.Module] = None,
        name_filter: List[str] = None,
    ):
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
            yield name, module

    def compute_lora_hessian(
        self,
        model: nn.Module,
        type_filter: List[nn.Module],
        name_filter: List[str],
    ):
        """
        Analytically computed compressed covariances using projection matrices(LoRA weights).
        """
        for name, _ in self._named_modules(model, type_filter, name_filter):
            parent, target, target_name = _get_submodules(model, name)
            if "lora_A" in target_name:
                mode = FORWARD
            elif "lora_C" in target_name:
                mode = BACKWARD
            else:
                continue
            projection_matrix = target.weight.data
            if mode == BACKWARD:
                projection_matrix = projection_matrix.t()
            hessian_state = self._state.get_hessian_state()
            orig_name = name.replace("." + target_name, "")
            new_name = orig_name + ".analog_lora_B"  # FIXME: do this with analog.watch
            orig_cov = hessian_state[orig_name][mode]
            projected_cov = (
                projection_matrix
                @ orig_cov.to(projection_matrix.device)
                @ projection_matrix.t()
            ).to(orig_cov.device) / 2
            self._state.hessian_state[new_name][mode] = projected_cov
        keys = list(self._state.hessian_state.keys())
        for k in keys:  # FIXME: do this with analog.clear
            if "analog_lora_B" not in k:
                del self._state.hessian_state[k]
        self._state._states_to_normalize.remove(
            ("hessian_state", "hessian_counter")
        )  # FIXME: do this with analog.finalize -> state.finalize

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
        for name, module in self._named_modules(model, type_filter, name_filter):
            lora_cls = None
            if isinstance(module, nn.Linear):
                lora_cls = LoraLinear
            elif isinstance(module, nn.Conv1d):
                raise NotImplementedError
            elif isinstance(module, nn.Conv2d):
                lora_cls = LoraConv2d

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
                lora_module.pca_init_weight(self.init_strategy, hessian_state[name])
            lora_module.to(device)

            parent, target, target_name = _get_submodules(model, name)
            setattr(parent, target_name, lora_module)

        self.compute_lora_hessian(model, type_filter, name_filter)
