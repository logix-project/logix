from typing import Optional

import torch
import torch.nn as nn

from logix.batch_info import BatchInfo
from logix.state import LogIXState
from logix.statistic.utils import make_2d


class Mean:
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
        mean_state = state.mean_state
        mean_counter = state.mean_counter
        if data is None:
            data = binfo.log[module_name][log_type]

        # extract and reshape data to 2d tensor for mean computation
        batch_size = data.size(0)
        data = make_2d(data, module, log_type).detach()

        # initialize mean state if necessary
        if log_type not in mean_state[module_name]:
            device = data.device if not cpu_offload else "cpu"
            dtype = data.dtype
            mean_state[module_name][log_type] = torch.zeros(
                data.shape[-1], device=device, dtype=dtype
            )
            mean_counter[module_name][log_type] = 0

        # update mean state
        if cpu_offload:
            # By default, all states are stored on the CPU, and therefore
            # computing updates for states on CPU is slow. For efficiency,
            # we move states to the GPU if data is on the GPU, and then
            # move it back to the CPU asynchrously.
            mean_state_gpu = mean_state[module_name][log_type].to(device=data.device)
            mean_state_gpu.add_(data.sum(dim=0))
            mean_state[module_name][log_type] = mean_state_gpu.to(
                device="cpu", non_blocking=True
            )
        else:
            mean_state[module_name][log_type].add_(data.sum(dim=0))

        # update mean counter
        if binfo.mask is None or log_type == "grad":
            mean_counter[module_name][log_type] += batch_size
        else:
            mean_counter[module_name][log_type] += binfo.mask.sum().item()
