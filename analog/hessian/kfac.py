from typing import Dict, Any

import torch
import torch.nn as nn

from analog.constants import FORWARD, BACKWARD
from analog.state import AnaLogState
from analog.hessian.base import HessianHandlerBase
from analog.hessian.utils import extract_actvations_expand, extract_actvations_reduce


class KFACHessianHandler(HessianHandlerBase):
    """
    Compute the Hessian via the K-FAC method.
    """

    def __init__(self, config: Dict[str, Any], state: AnaLogState) -> None:
        super().__init__(config, state)
        self.ekfac = False

    def parse_config(self) -> None:
        self.damping = self.config.get("damping", 1e-2)
        self.reduce = self.config.get("reduce", False)

    @torch.no_grad()
    def on_exit(self, log_state=None) -> None:
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
            for module_name, module_state in log_state.items():
                for mode, data in module_state.items():
                    self.update_hessian_reduce(None, module_name, mode, data)

        if self.ekfac:
            for module_name, module_grad in log_state.items():
                self.update_ekfac(module_name, module_grad)

    def update_hessian_reduce(
        self,
        module: nn.Module,
        module_name: str,
        mode: str,
        data: torch.Tensor,
    ):
        if not self.reduce or self.ekfac:
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
        hessian_state = self._state.hessian_state
        sample_counter = self._state.hessian_counter

        # initialize hessian state if necessary
        if mode not in hessian_state[module_name]:
            hessian_state[module_name][mode] = torch.zeros(
                (activations.shape[-1], activations.shape[-1])
            )
            sample_counter[module_name][mode] = 0

        # compute covariance and update hessian state
        if activations.is_cuda:
            # By default, the hessian state is stored on the CPU. However,
            # computing/updating the hessian state is on CPU is slow. Thus,
            # we move the hessian state to the GPU if the activation is on
            # the GPU, and then move it back to the CPU asynchronously.
            hessian_state_gpu = hessian_state[module_name][mode].to(
                device=activations.device
            )
            hessian_state_gpu.addmm_(activations.t(), activations)
            hessian_state[module_name][mode] = hessian_state_gpu.to(
                device="cpu", non_blocking=True
            )
        else:
            hessian_state[module_name][mode].addmm_(activations.t(), activations)

        sample_counter[module_name][mode] += len(activations)

    @torch.no_grad()
    def update_ekfac(
        self,
        module_name: str,
        data: torch.Tensor,
    ) -> None:
        hessian_eigvec_state = self._state.hessian_eigvec_state
        hessian_eigval_state = self._state.hessian_eigval_state
        ekfac_eigval_state = self._state.ekfac_eigval_state
        ekfac_counter = self._state.ekfac_counter

        if module_name not in ekfac_eigval_state:
            ekfac_eigval_state[module_name] = torch.zeros(
                data.shape[-2], data.shape[-1]
            )
            ekfac_counter[module_name] = 0

        data = data.cpu().detach()
        rotated_grads = torch.matmul(data, hessian_eigvec_state[module_name][FORWARD])
        for rotated_grad in rotated_grads:
            weight = torch.matmul(
                hessian_eigvec_state[module_name][BACKWARD].t(), rotated_grad
            )
            ekfac_eigval_state[module_name].add_(torch.square(weight), alpha=1.0)
        ekfac_counter[module_name] += len(data)
