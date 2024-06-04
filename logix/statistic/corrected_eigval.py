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


class CorrectedEigval:
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
        Update the mean state.
        """
        if not hasattr(state, "covariance_eigvec_state"):
            state.covariance_svd()
            state.register_state("ekfac_eigval_state", synchronize=True, save=True)
            state.register_state("ekfac_counter", synchronize=True, save=False)
            state.register_normalize_pair("ekfac_eigval_state", "ekfac_counter")
        covariance_eigvec_state = state.covariance_eigvec_state
        covariance_eigval_state = state.covariance_eigval_state
        ekfac_eigval_state = state.ekfac_eigval_state
        ekfac_counter = state.ekfac_counter

        assert data is None
        data = binfo.log[module_name]["grad"]

        if module_name not in ekfac_eigval_state:
            device = data.device if not cpu_offload else "cpu"
            dtype = data.dtype
            ekfac_eigval_state[module_name] = torch.zeros(
                data.shape[-2], data.shape[-1], device=device, dtype=dtype
            )
            ekfac_counter[module_name] = 0

        data = data.detach()
        if cpu_offload:
            eigvec_fwd_gpu = covariance_eigvec_state[module_name]["forward"].to(
                device=data.device
            )
            eigvec_bwd_gpu = covariance_eigvec_state[module_name]["backward"].to(
                device=data.device
            )
            ekfac_eigval_state_gpu = ekfac_eigval_state[module_name].to(
                device=data.device
            )
            rotated_grads = torch.matmul(data, eigvec_fwd_gpu)
            for rotated_grad in rotated_grads:
                weight = torch.matmul(eigvec_bwd_gpu.t(), rotated_grad)
                ekfac_eigval_state_gpu.add_(weight.square_())
            ekfac_eigval_state[module_name] = ekfac_eigval_state_gpu.to(
                device="cpu", non_blocking=True
            )

            # TODO: Not sure if this improves memory usage.
            del eigvec_fwd_gpu
            del eigvec_bwd_gpu
            del ekfac_eigval_state_gpu
        else:
            rotated_grads = torch.matmul(
                data, covariance_eigvec_state[module_name]["forward"]
            )
            for rotated_grad in rotated_grads:
                weight = torch.matmul(
                    covariance_eigvec_state[module_name]["backward"].t(), rotated_grad
                )
                ekfac_eigval_state[module_name].add_(weight.square_())

        ekfac_counter[module_name] += len(data)
