# Copyright 2023-present the LogIX team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import torch.nn as nn

from logix.config import LoRAConfig
from logix.lora.modules import LoraConv2d, LoraEmbedding, LoraLinear
from logix.lora.utils import _get_submodules, find_parameter_sharing_group
from logix.state import LogIXState
from logix.utils import get_logger, module_check


class LoRAHandler:
    """
    Transforms a model into a Lora model.
    """

    _SUPPORTED_MODULES = {nn.Linear, nn.Conv1d, nn.Conv2d}

    def __init__(self, config: LoRAConfig, state: LogIXState):
        self._state = state

        self.init_strategy = config.init
        self.rank = config.rank
        self.parameter_sharing = config.parameter_sharing
        self.parameter_sharing_groups = config.parameter_sharing_groups

    def add_lora(
        self,
        model: nn.Module,
        type_filter: Optional[List[nn.Module]],
        name_filter: Optional[List[str]],
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
