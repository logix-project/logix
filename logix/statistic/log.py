from typing import Optional

import torch
import torch.nn as nn

from logix.batch_info import BatchInfo
from logix.state import LogIXState


class Log:
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
        Put log into `binfo`
        """
        module_log = binfo[module_name]
        if log_type not in module_log:
            module_log[log_type] = data
        else:
            module_log[log_type] += data
