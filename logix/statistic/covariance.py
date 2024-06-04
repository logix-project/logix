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

from typing import Optional

import torch
import torch.nn as nn

from logix.batch_info import BatchInfo
from logix.state import LogIXState
from logix.statistic.utils import make_2d


class Covariance:
    @staticmethod
    @torch.no_grad()
    def update(
        state: LogIXState,
        binfo: BatchInfo,
        module: nn.Module,
        module_name: str,
        log_type: str,
        data: Optional[torch.Tensor] = None,
        cpu_offload: Optional[bool] = False,
    ):
        """
        Update the covariance state.
        """
        covariance_state = state.covariance_state
        covariance_counter = state.covariance_counter
        if data is None:
            data = binfo.log[module_name][log_type]

        # extract and reshape data to 2d tensor for mean computation
        batch_size = data.size(0)
        data = make_2d(data, module, log_type).detach()

        # initialize covariance state if necessary
        if log_type not in covariance_state[module_name]:
            device = data.device if not cpu_offload else "cpu"
            dtype = data.dtype
            covariance_state[module_name][log_type] = torch.zeros(
                data.shape[-1], data.shape[-1], device=device, dtype=dtype
            )
            covariance_counter[module_name][log_type] = 0

        # update mean state
        if cpu_offload:
            # By default, all states are stored on the CPU, and therefore
            # computing updates for states on CPU is slow. For efficiency,
            # we move states to the GPU if data is on the GPU, and then
            # move it back to the CPU asynchrously.
            covariance_state_gpu = covariance_state[module_name][log_type].to(
                device=data.device
            )
            covariance_state_gpu.addmm_(data.t(), data)
            covariance_state[module_name][log_type] = covariance_state_gpu.to(
                device="cpu", non_blocking=True
            )
        else:
            covariance_state[module_name][log_type].addmm_(data.t(), data)

        # update mean counter
        if binfo.mask is None or log_type == "grad":
            covariance_counter[module_name][log_type] += batch_size
        else:
            covariance_counter[module_name][log_type] += binfo.mask.sum().item()
