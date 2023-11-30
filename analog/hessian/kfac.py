import os
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
        self.log_dir = self.config.get("log_dir")
        self.damping = self.config.get("damping", 1e-2)
        self.reduce = self.config.get("reduce", False)

    @torch.no_grad()
    def on_exit(self, current_log=None, update_hessian=True) -> None:
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
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
        activation = self.extract_activations(module, mode, data).detach()

        # update covariance
        if deep_get(self.hessian_state, [module_name, mode]) is None:
            self.hessian_state[module_name][mode] = torch.zeros(
                (activation.shape[-1], activation.shape[-1])
            )
            self.sample_counter[module_name][mode] = 0

        # move to gpu
        if activation.is_cuda:
            hessian_state_gpu = self.hessian_state[module_name][mode].to(
                device=activation.device
            )
            hessian_state_gpu.addmm_(activation.t(), activation)
            self.hessian_state[module_name][mode] = hessian_state_gpu.to(
                device="cpu", non_blocking=True
            )
        else:
            self.hessian_state[module_name][mode].addmm_(activation.t(), activation)
        self.sample_counter[module_name][mode] += len(data)

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
                ekfac_eigval.div_(get_world_size())
                dist.all_reduce(ekfac_eigval, op=dist.ReduceOp.SUM)
        else:
            for _, module_state in self.hessian_state.items():
                for _, covariance in module_state.items():
                    covariance.div_(get_world_size())
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

    def save_state(self) -> None:
        """
        Save Hessian state to disk.
        """
        # TODO: should this be in the constructor or initialize-type function?
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # TODO: implement this for all HessianHandlers
        if hasattr(self, "hessian_state"):
            torch.save(self.hessian_state, os.path.join(self.log_dir, "hessian.pt"))
        if hasattr(self, "hessian_eigvec_state"):
            torch.save(
                self.hessian_eigvec_state,
                os.path.join(self.log_dir, "hessian_eigvec.pt"),
            )
        if hasattr(self, "hessian_eigval_state"):
            torch.save(
                self.hessian_eigval_state,
                os.path.join(self.log_dir, "hessian_eigval.pt"),
            )
        if hasattr(self, "ekfac_eigval_state"):
            torch.save(
                self.ekfac_eigval_state,
                os.path.join(self.log_dir, "ekfac_eigval.pt"),
            )
        if hasattr(self, "hessian_inverse_state"):
            torch.save(
                self.hessian_inverse_state,
                os.path.join(self.log_dir, "hessian_inverse.pt"),
            )

    def load_state(self, log_dir: str) -> None:
        """
        Load Hessian state from disk.
        """
        # TODO: implement this for all HessianHandlers
        assert os.path.exists(log_dir), "Hessian log directory does not exist!"
        log_dir_items = os.listdir(log_dir)
        if "hessian.pt" in log_dir_items:
            self.hessian_state = torch.load(os.path.join(log_dir, "hessian.pt"))
        if "hessian_eigvec.pt" in log_dir_items:
            self.hessian_eigvec_state = torch.load(
                os.path.join(log_dir, "hessian_eigvec.pt")
            )
        if "hessian_eigval.pt" in log_dir_items:
            self.hessian_eigval_state = torch.load(
                os.path.join(log_dir, "hessian_eigval.pt")
            )
        if "ekfac_eigval.pt" in log_dir_items:
            self.ekfac_eigval_state = torch.load(
                os.path.join(log_dir, "ekfac_eigval.pt")
            )
        if "hessian_inverse.pt" in log_dir_items:
            self.hessian_inverse_state = torch.load(
                os.path.join(log_dir, "hessian_inverse.pt")
            )
