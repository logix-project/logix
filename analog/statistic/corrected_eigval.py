from typing import Optional

import torch
import torch.nn as nn

from analog.batch_info import BatchInfo
from analog.statistic import StatisticState
from analog.statistic.base import StatisticComputerBase


class CorrectedEigval(StatisticComputerBase):
    def update(
        self,
        state: StatisticState,
        binfo: BatchInfo,
        module: nn.Module,
        module_name: str,
        log_type: str,
        data: Optional[torch.Tensor] = None,
    ):
        """
        Update the mean state.
        """
        covariance_eigvec_state = state.hessian_eigvec_state
        covariance_eigval_state = state.hessian_eigval_state
        ekfac_eigval_state = state.ekfac_eigval_state
        ekfac_counter = state.ekfac_counter

        assert data is None
        data = binfo.log[module_name]["grad"]

        if module_name not in ekfac_eigval_state:
            ekfac_eigval_state[module_name] = torch.zeros(
                data.shape[-2], data.shape[-1]
            )
            ekfac_counter[module_name] = 0

        data = data.detach()
        if data.is_cuda:
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
