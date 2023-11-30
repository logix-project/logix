from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from analog.utils import nested_dict


class HessianHandlerBase(ABC):
    def __init__(self, config) -> None:
        self.config = config
        self.hessian_state = nested_dict()
        self.sample_counter = nested_dict()

        self.hessian_inverse_with_override = False
        self.hessian_svd_with_override = False

        # Logging
        self.file_prefix = "hessian_log_"

        self.parse_config()

    @abstractmethod
    def parse_config(self) -> None:
        """
        Parse the configuration parameters.
        """
        pass

    @abstractmethod
    def update_hessian(
        self,
        module: nn.Module,
        module_name: str,
        mode: str,
        data: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> None:
        """
        Compute the covariance for given data.
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """
        Finalize the covariance computation.
        """
        pass

    @abstractmethod
    def on_exit(self, current_log=None) -> None:
        pass

    def get_hessian_state(self, name: str = None):
        """
        Get the Hessian state.
        """
        if name is None:
            return self.hessian_state
        assert name in self.hessian_state
        return self.hessian_state[name]

    def get_hessian_inverse_state(self, name: str = None):
        """
        Get the Hessian inverse state.
        """
        if name is None:
            return (
                self.hessian_state
                if self.hessian_inverse_with_override
                else self.hessian_inverse_state
            )
        return (
            self.hessian_state[name]
            if self.hessian_inverse_with_override
            else self.hessian_inverse_state[name]
        )

    def get_hessian_svd_state(self, name: str = None):
        """
        Get the Hessian SVD state.
        """
        if name is None:
            return (
                self.hessian_state
                if self.hessian_svd_with_override
                else self.hessian_svd_state
            )
        return (
            self.hessian_state[name]
            if self.hessian_svd_with_override
            else self.hessian_svd_state[name]
        )

    def get_sample_size(
        self, data: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> int:
        """
        Get the sample size for the given data.
        """
        if mask is not None:
            return mask.sum().item()
        return data.size(0)

    def clear(self) -> None:
        """
        Clear the Hessian state.
        """
        self.hessian_state.clear()
        if hasattr(self, "hessian_inverse_state"):
            del self.hessian_inverse_state
        self.sample_counter.clear()
