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
        self.log_dir = self.config.get("log_dir")
        self.damping = self.config.get("damping", 1e-2)
        self.reduce = self.config.get("reduce", True)

    @torch.no_grad()
    def on_exit(self, current_log=None) -> None:
        if self.reduce:
            for module_name, module_grad in current_log.items():
                flat_grad = rearrange(module_grad, "b ... -> b (...)").cpu().detach()
                grad_dim = flat_grad.shape[-1]
                if module_name not in self.hessian_state:
                    self.hessian_state[module_name] = torch.zeros(
                        (grad_dim, grad_dim), device="cpu"
                    )
                    self.sample_counter[module_name] = 0
                self.hessian_state[module_name].addmm_(flat_grad.t(), flat_grad)
                self.sample_counter[module_name] += flat_grad.shape[0]

    @torch.no_grad()
    def update_hessian(
        self,
        module: nn.Module,
        module_name: str,
        mode: str,
        data: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.reduce:
            flat_grad = rearrange(data, "b ... -> b (...)").cpu().detach()
            grad_dim = flat_grad.shape[-1]

            # update covariance
            if module_name not in self.hessian_state:
                self.hessian_state[module_name] = torch.zeros(
                    (grad_dim, grad_dim), device="cpu"
                )
                self.sample_counter[module_name] = 0
            self.hessian_state[module_name].addmm_(flat_grad.t(), flat_grad)
            self.sample_counter[module_name] += data.shape[0]
