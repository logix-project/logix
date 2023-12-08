from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from analog.state import AnaLogState
from analog.utils import nested_dict


class HessianHandlerBase(ABC):
    def __init__(self, config: Dict[str, Any], state: AnaLogState) -> None:
        self.config = config
        self._state = state

        self.parse_config()

    @abstractmethod
    def parse_config(self) -> None:
        """
        Parse the configuration parameters.
        """
        pass

    @abstractmethod
    def on_exit(self, current_log=None) -> None:
        pass

    def get_sample_size(
        self, data: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> int:
        """
        Get the sample size for the given data.
        """
        if mask is not None:
            return mask.sum().item()
        return data.size(0)
