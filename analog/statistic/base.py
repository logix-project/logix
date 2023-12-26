from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from analog.batch_info import BatchInfo
from analog.statistic import StatisticState


class StatisticComputerBase(ABC):
    def __call__(
        self,
        state: StatisticState,
        binfo: BatchInfo,
        module: nn.Module,
        module_name: str,
        log_type: str,
        data: Optional[torch.Tensor] = None,
    ):
        self.update(state, binfo, module, module_name, log_type, data)

    @abstractmethod
    def update(
        self,
        state: StatisticState,
        binfo: BatchInfo,
        module: nn.Module,
        module_name: str,
        log_type: str,
        data: Optional[torch.Tensor] = None,
    ):
        pass
