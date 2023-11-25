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

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.ekfac = False

    def parse_config(self) -> None:
        self.damping = self.config.get("damping", 1e-2)
        self.reduce = self.config.get("reduce", False)

    @torch.no_grad()
    def on_exit(self, current_log=None, update_hessian=True) -> None:
        if update_hessian:
            if self.reduce:
                raise NotImplementedError

            if self.ekfac:
                for module_name, module_grad in current_log.items():
                    self.update_ekfac(module_name, module_grad)

    @torch.no_grad()
    def update_hessian(
        self,
        module: nn.Module,
        module_name: str,
        mode: str,
        data: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if self.reduce or self.ekfac:
            return
        # extract activations
        activation = self.extract_activations(module, mode, data)

        # compute covariance
        covariance = torch.matmul(torch.t(activation), activation).cpu().detach()

        # update covariance
        if deep_get(self.hessian_state, [module_name, mode]) is None:
            self.hessian_state[module_name][mode] = torch.zeros_like(covariance)
            self.sample_counter[module_name][mode] = 0
        self.hessian_state[module_name][mode].add_(covariance)
        self.sample_counter[module_name][mode] += self.get_sample_size(data, mask)

    @torch.no_grad()
    def update_ekfac(
        self,
        module_name: str,
        data: torch.Tensor,
    ) -> None:
        if not hasattr(self, "hessian_eigval_state"):
            self.hessian_svd(set_attr=True)
        if not hasattr(self, "ekfac_eigval_state"):
            self.ekfac_eigval_state = nested_dict()
            self.ekfac_counter = nested_dict()

        data = data.cpu().detach()
        if module_name not in self.ekfac_eigval_state:
            self.ekfac_eigval_state[module_name] = torch.zeros(
                data.shape[-2], data.shape[-1]
            )
            self.ekfac_counter[module_name] = 0

        self.ekfac_counter[module_name] += len(data)
        rotated_grads = torch.matmul(
            data, self.hessian_eigvec_state[module_name][FORWARD]
        )
        for rotated_grad in rotated_grads:
            weight = torch.matmul(
                self.hessian_eigvec_state[module_name][BACKWARD].t(), rotated_grad
            )
            self.ekfac_eigval_state[module_name].add_(torch.square(weight), alpha=0.5)

    def finalize(self) -> None:
        if self.ekfac:
            for module_name, ekfac_eigval in self.ekfac_eigval_state.items():
                ekfac_eigval.div_(self.ekfac_counter[module_name])
        else:
            for module_name, module_state in self.hessian_state.items():
                for mode, covariance in module_state.items():
                    covariance.div_(self.sample_counter[module_name][mode])

        self.synchronize()

    @torch.no_grad()
    def hessian_inverse(self, set_attr: bool = False):
        """
        Compute the inverse of the covariance.
        """
        hessian_inverse_state = nested_dict()
        for module_name, module_state in self.hessian_state.items():
            for mode, covariance in module_state.items():
                hessian_inverse_state[module_name][mode] = torch.inverse(
                    covariance
                    + torch.trace(covariance)
                    * self.damping
                    * torch.eye(covariance.size(0))
                    / covariance.size(0)
                )

        if set_attr:
            self.hessian_inverse_state = hessian_inverse_state

        return hessian_inverse_state

    @torch.no_grad()
    def hessian_svd(self, set_attr: bool = False):
        """
        Compute the SVD of the covariance.
        """
        hessian_eigval_state = nested_dict()
        hessian_eigvec_state = nested_dict()
        for module_name, module_state in self.hessian_state.items():
            for mode, covariance in module_state.items():
                eigvals, eigvecs = torch.linalg.eigh(covariance)
                hessian_eigval_state[module_name][mode] = eigvals
                hessian_eigvec_state[module_name][mode] = eigvecs

        if set_attr:
            self.hessian_eigval_state = hessian_eigval_state
            self.hessian_eigvec_state = hessian_eigvec_state

        return hessian_eigval_state, hessian_eigvec_state

    def get_hessian_inverse_state(self):
        if not hasattr(self, "hessian_inverse_state"):
            self.hessian_inverse(set_attr=True)
        return self.hessian_inverse_state

    def get_hessian_svd_state(self):
        if not hasattr(self, "hessian_eigval_state"):
            self.hessian_svd(set_attr=True)
        if self.ekfac and hasattr(self, "ekfac_eigval_state"):
            return self.ekfac_eigval_state, self.hessian_eigvec_state, True
        return self.hessian_eigval_state, self.hessian_eigvec_state, False

    def synchronize(self) -> None:
        """
        Synchronize the covariance across all processes.
        """
        if get_world_size() <= 1:
            return

        if self.ekfac:
            for _, ekfac_eigval in self.ekfac_eigval_state.items():
                ekfac_eigval.div_(world_size)
                dist.all_reduce(ekfac_eigval, op=dist.ReduceOp.SUM)
        else:
            for _, module_state in self.hessian_state.items():
                for _, covariance in module_state.items():
                    covariance.div_(world_size)
                    dist.all_reduce(covariance, op=dist.ReduceOp.SUM)

    def clear(self) -> None:
        """
        Clear the Hessian state.
        """
        super().clear()
        if hasattr(self, "hessian_eigval_state"):
            del self.hessian_eigval_state
            del self.hessian_eigvec_state
        if hasattr(self, "ekfac_eigval_state"):
            del self.ekfac_eigval_state
            del self.ekfac_counter

    def extract_activations(
        self,
        module: nn.Module,
        mode: str,
        data: torch.Tensor,
    ) -> torch.Tensor:
        if mode == FORWARD:
            return extract_forward_activations(data, module)
        assert mode == BACKWARD
        return extract_backward_activations(data, module)
