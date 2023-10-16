from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from analog.utils import deep_get, get_world_size
from analog.hessian.base import HessianHandlerBase
from analog.hessian.utils import extract_activations, extract_gradients


class KFACHessianHandler(HessianHandlerBase):
    """
    Compute the Hessian via the K-FAC method.
    """

    def update_hessian(
        self,
        module: nn.Module,
        module_name: str,
        mode: str,
        data: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        extract_fn = extract_activations if mode == "forward" else extract_gradients
        activation = extract_fn(data, module, mask)
        covariance = torch.matmul(torch.t(activation), activation).cpu().detach()
        if deep_get(self.hessian_state, [module_name, mode]) is None:
            self.hessian_state[module_name][mode] = covariance
            self.sample_counter[module_name][mode] = data.size(0)
        else:
            self.hessian_state[module_name][mode] += covariance
            self.sample_counter[module_name][mode] += data.size(0)

    def finalize(self):
        for module_name, module_state in self.hessian_state.items():
            for mode, covariance in module_state.items():
                covariance.div_(self.sample_counter[module_name][mode])
        self.synchronize()
        self.hessian_inverse()

    def hessian_inverse(self):
        """
        Compute the inverse of the covariance.
        """
        for _, module_state in self.hessian_state.items():
            for mode, covariance in module_state.items():
                module_state[mode] = torch.inverse(
                    covariance + torch.trace(covariance) * self.config.damping
                )

    def synchronize(self):
        """
        Synchronize the covariance across all processes.
        """
        world_size = get_world_size()
        if world_size > 1:
            for _, module_state in self.hessian_state.items():
                for _, covariance in module_state.items():
                    covariance.div_(world_size)
                    dist.all_reduce(covariance, op=dist.ReduceOp.SUM)
