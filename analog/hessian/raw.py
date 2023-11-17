from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from einops import rearrange

from analog.constants import FORWARD, BACKWARD
from analog.utils import deep_get, get_world_size, nested_dict, get_logger
from analog.hessian.base import HessianHandlerBase
from analog.hessian.utils import (
    extract_forward_activations,
    extract_backward_activations,
)


class RawHessianHandler(HessianHandlerBase):
    """
    Compute the Hessian via direct gradient outer-product.
    """

    def parse_config(self) -> None:
        self.damping = self.config.get("damping", 1e-2)
        self.type = self.config.get("type", "reduce")

    def on_exit(self) -> None:
        if self.type == "reduce":
            cur_log = None
            for module_name, module_grad in cur_log.items():
                flat_grad = rearrange(data, "b ... -> b (...)").cpu().detach()
                grad_dim = flat_grad.shape[-1]
            
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
            flat_grad = rearrange(data, "b ... -> b (...)").cpu().detach()
            grad_dim = flat_grad.shape[-1]

            # update covariance
            if module_name not in self.hessian_state:
                self.hessian_state[module_name] = torch.zeros_like(covariance)
                self.sample_counter[module_name] = 0
            self.hessian_state[module_name].addmm_(flat_grad.t(), flat_grad)
            self.sample_counter[module_name] += data.shape[0]

    def finalize(self) -> None:
        for module_name, module_hessian in self.hessian_state.items():
            module_hessian.div_(self.sample_counter[module_name])
        self.synchronize()

    @torch.no_grad()
    def hessian_inverse(self, override: bool = False) -> None:
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
        for module_name, module_hessian in self.hessian_state.items():
            if override:
                self.hessian_state[module_name] = torch.inverse(
                    module_hessian
                    + torch.trace(module_hessian)
                    * self.damping
                    * torch.eye(module_hessian.size(0))
                )
            else:
                self.hessian_inverse_state[module_name] = torch.inverse(
                    module_hessian
                    + torch.trace(module_hessian)
                    * self.damping
                    * torch.eye(module_hessian.size(0))
                )

    def synchronize(self) -> None:
        """
        Synchronize the covariance across all processes.
        """
        world_size = get_world_size()
        if world_size > 1:
            for _, module_hessian in self.hessian_state.items():
                module_hessian.div_(world_size)
                dist.all_reduce(module_hessian, op=dist.ReduceOp.SUM)
