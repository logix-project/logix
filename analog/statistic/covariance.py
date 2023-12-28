from typing import Optional

import torch
import torch.nn as nn

from analog.batch_info import BatchInfo
from analog.state import StatisticState
from analog.statistic.utils import make_2d


class Covariance:
    @staticmethod
    @torch.no_grad()
    def update(
        state: StatisticState,
        binfo: BatchInfo,
        module: nn.Module,
        module_name: str,
        log_type: str,
        data: Optional[torch.Tensor] = None,
    ):
        """
        Update the covariance state.
        """
        covariance_state = state.covariance_state
        covariance_counter = state.covariance_counter
        if data is None:
            data = binfo.log[module_name][log_type]

        # initialize covariance state if necessary
        if log_type not in covariance_state[module_name]:
            covariance_state[module_name][log_type] = torch.zeros(
                data.shape[-1], data.shape[-1]
            )
            covariance_counter[module_name][log_type] = 0

        # extract and reshape data to 2d tensor for mean computation
        data = make_2d(data, module, log_type).detach()

        # update mean state
        if data.is_cuda:
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
        if binfo.mask is None:
            covariance_counter[module_name][log_type] += len(data)
        else:
            covariance_counter[module_name][log_type] += binfo.mask.sum().item()
