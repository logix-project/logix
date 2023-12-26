from typing import Optional

import torch
import torch.nn as nn

from analog.batch_info import BatchInfo
from analog.statistic import StatisticState
from analog.statistic.base import StatisticComputerBase
from analog.statistic.utils import make_2d


class Mean(StatisticComputerBase):
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
        mean_state = state.mean_state[module_name][log_type]
        mean_counter = state.mean_counter[module_name][log_type]
        if data is None:
            data = binfo.log[module_name][log_type]

        # initialize mean state if necessary
        if not isinstance(mean_state, torch.Tensor):
            assert isinstance(mean_state, dict)
            mean_state[module_name][log_type] = torch.zeros(data.shape[-1])
            mean_counter[module_name][log_type] = 0

        # extract and reshape data to 2d tensor for mean computation
        data = make_2d(data, module, log_type)

        # update mean state
        if data.is_cuda:
            # By default, all states are stored on the CPU, and therefore
            # computing updates for states on CPU is slow. For efficiency,
            # we move states to the GPU if data is on the GPU, and then
            # move it back to the CPU asynchrously.
            mean_state_gpu = mean_state[module_name][log_type].to(device=data.device)
            mean_state_gpu.add_(data.sum(dim=0))
            mean_state[module_name] = mean_state_gpu.to(device="cpu", non_blocking=True)
        else:
            mean_state[module_name] += data

        # update mean counter
        if binfo.mask is None:
            mean_counter[module_name] += len(data)
        else:
            mean_counter[module_name] += binfo.mask.sum().item()
