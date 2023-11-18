from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from analog.constants import FORWARD, BACKWARD
from analog.utils import deep_get, get_world_size, nested_dict, get_logger
from analog.hessian.base import HessianHandlerBase
from analog.hessian.utils import (
    extract_forward_activations,
    extract_backward_activations,
)


class KFACHessianHandler(HessianHandlerBase):
    """
    Compute the Hessian via the K-FAC method.
    """

    def parse_config(self) -> None:
        self.damping = self.config.get("damping", 1e-2)
        self.type = self.config.get("type", "expand")

    @torch.no_grad()
    def on_exit(self, current_log=None) -> None:
        if self.type == "reduce":
            raise NotImplementedError

    @torch.no_grad()
    def update_hessian(
        self,
        module: nn.Module,
        module_name: str,
        mode: str,
        data: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if self.type == "expand":
            # extract activations
            activation = self.extract_activations(module, mode, data, mask)

            # compute covariance
            covariance = torch.matmul(torch.t(activation), activation).cpu().detach()

            # update covariance
            if deep_get(self.hessian_state, [module_name, mode]) is None:
                self.hessian_state[module_name][mode] = torch.zeros_like(covariance)
                self.sample_counter[module_name][mode] = 0
            self.hessian_state[module_name][mode].add_(covariance)
            self.sample_counter[module_name][mode] += self.get_sample_size(data, mask)

    def finalize(self) -> None:
        for module_name, module_state in self.hessian_state.items():
            for mode, covariance in module_state.items():
                covariance.div_(self.sample_counter[module_name][mode])
        self.synchronize()

    @torch.no_grad()
    def hessian_inverse(self, override: bool = False):
        """
        Compute the inverse of the covariance.
        """
        if self.hessian_inverse_with_override:
            get_logger().warning("Hessian inverse already computed with override.")
            return

        if override:
            self.hessian_inverse_with_override = True
        else:
            self.hessian_inverse_state = nested_dict()
        for module_name, module_state in self.hessian_state.items():
            for mode, covariance in module_state.items():
                if override:
                    module_state[mode] = torch.inverse(
                        covariance
                        + torch.trace(covariance)
                        * self.damping
                        * torch.eye(covariance.size(0))
                    )
                else:
                    self.hessian_inverse_state[module_name][mode] = torch.inverse(
                        covariance
                        + torch.trace(covariance)
                        * self.damping
                        * torch.eye(covariance.size(0))
                    )

    def synchronize(self) -> None:
        """
        Synchronize the covariance across all processes.
        """
        world_size = get_world_size()
        if world_size > 1:
            for _, module_state in self.hessian_state.items():
                for _, covariance in module_state.items():
                    covariance.div_(world_size)
                    dist.all_reduce(covariance, op=dist.ReduceOp.SUM)

    def extract_activations(
        self,
        module: nn.Module,
        mode: str,
        data: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mode == FORWARD:
            return extract_forward_activations(data, module, mask)
        assert mode == BACKWARD
        return extract_backward_activations(data, module, mask)
