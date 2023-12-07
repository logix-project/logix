import os
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from analog.constants import FORWARD, BACKWARD
from analog.utils import get_world_size, nested_dict, get_rank
from analog.hessian.base import HessianHandlerBase
from analog.hessian.utils import extract_actvations_expand, extract_actvations_reduce


class KFACHessianHandler(HessianHandlerBase):
    """
    Compute the Hessian via the K-FAC method.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.ekfac = False

        self.hessian_state_unsync = False
        self.ekfac_state_unsync = False

    def parse_config(self) -> None:
        self.log_dir = self.config.get("log_dir")
        self.damping = self.config.get("damping", 1e-2)
        self.reduce = self.config.get("reduce", False)

    @torch.no_grad()
    def on_exit(self, current_log) -> None:
        """
        This function is called when the code is exiting the AnaLog context.
        Given the analogy between Hessian state and traiditional optimizer
        state, this function is analogous to the optimizer's step function.

        Args:
            current_log (optional): The current log. If not specified, the
                Hessian state will not be updated.
            update_hessian (optional): Whether to update the Hessian state.
        """
        if self.reduce and not self.ekfac:
            for module_name, module_state in current_log.items():
                for mode, data in module_state.items():
                    self.update_hessian_reduce(None, module_name, mode, data)

        if self.ekfac:
            for module_name, module_grad in current_log.items():
                self.update_ekfac(module_name, module_grad)

    def update_hessian_reduce(
        self,
        module: nn.Module,
        module_name: str,
        mode: str,
        data: torch.Tensor,
    ):
        if self.expand or self.ekfac:
            return

        # extract activations
        activations = extract_actvations_reduce(module, mode, data).detach()

        # update hessian
        self.update_hessian(module_name, mode, activations)

    def update_hessian_expand(
        self,
        module: nn.Module,
        module_name: str,
        mode: str,
        data: torch.Tensor,
    ):
        if self.reduce or self.ekfac:
            return

        # extract activations
        activations = extract_actvations_expand(module, mode, data).detach()

        # update hessian
        self.update_hessian(module_name, mode, activations)

    @torch.no_grad()
    def update_hessian(
        self,
        module_name: str,
        mode: str,
        activations: torch.Tensor,
    ) -> None:
        """
        Update the Hessian state given a module and data. In KFAC, the Hessian
        state is divided into two parts: the forward and backward covariance.

        Args:
            module_name: The name of the module.
            mode: The mode of the module.
            data: The input data.
        """
        # initialize hessian state if necessary
        if mode not in self.hessian_state[module_name]:
            self.hessian_state[module_name][mode] = torch.zeros(
                (activations.shape[-1], activations.shape[-1])
            )
            self.sample_counter[module_name][mode] = 0

        # compute covariance and update hessian state
        if activations.is_cuda:
            # By default, the hessian state is stored on the CPU. However,
            # computing/updating the hessian state is on CPU is slow. Thus,
            # we move the hessian state to the GPU if the activation is on
            # the GPU, and then move it back to the CPU asynchrously.
            hessian_state_gpu = self.hessian_state[module_name][mode].to(
                device=activations.device
            )
            hessian_state_gpu.addmm_(activations.t(), activations)
            self.hessian_state[module_name][mode] = hessian_state_gpu.to(
                device="cpu", non_blocking=True
            )
        else:
            self.hessian_state[module_name][mode].addmm_(activations.t(), activations)

        self.sample_counter[module_name][mode] += len(activations)

        self.hessian_state_unsync = True

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

        if module_name not in self.ekfac_eigval_state:
            self.ekfac_eigval_state[module_name] = torch.zeros(
                data.shape[-2], data.shape[-1]
            )
            self.ekfac_counter[module_name] = 0

        data = data.cpu().detach()
        rotated_grads = torch.matmul(
            data, self.hessian_eigvec_state[module_name][FORWARD]
        )
        for rotated_grad in rotated_grads:
            weight = torch.matmul(
                self.hessian_eigvec_state[module_name][BACKWARD].t(), rotated_grad
            )
            self.ekfac_eigval_state[module_name].add_(torch.square(weight), alpha=0.5)
        self.ekfac_counter[module_name] += len(data)

        self.ekfac_state_unsync = True

    def finalize(self) -> None:
        """
        Finalize the Hessian state by synchronizing the state across processes.
        """
        if hasattr(self, "hessian_state") and self.hessian_state_unsync:
            self.synchronize(self.hessian_state, self.sample_counter)
            self.hessian_state_unsync = False
        if hasattr(self, "hessian_eigval_state") and self.ekfac_state_unsync:
            self.synchronize(self.ekfac_eigval_state, self.ekfac_counter)
            self.ekfac_state_unsync = False

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

    def synchronize(self, state_dict, counter_dict):
        """
        Synchronize the Hessian state across processes.
        """
        world_size = get_world_size()

        def _synchronize(state_dict, counter_dict):
            for key in state_dict:
                assert key in counter_dict
                if not isinstance(state_dict[key], torch.Tensor):
                    _synchronize(state_dict[key], counter_dict[key])
                else:
                    state_dict[key].div_(counter_dict[key])
                    if world_size > 1:
                        state_gpu = state_dict[key].cuda()
                        state_gpu.div_(world_size)
                        dist.all_reduce(state_gpu, op=dist.ReduceOp.SUM)
                        state_dict[key].copy_(state_gpu.cpu())

        _synchronize(state_dict, counter_dict)

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
