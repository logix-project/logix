import torch
from einops import rearrange

from analog.hessian.base import HessianHandlerBase


class RawHessianHandler(HessianHandlerBase):
    """
    Compute the Hessian via direct gradient outer-product.
    """

    def parse_config(self) -> None:
        self.damping = self.config.get("damping", 1e-2)

    @torch.no_grad()
    def on_exit(self, log_state=None) -> None:
        for module_name, module_grad in log_state:
            self.update_hessian(module_name, module_grad)

    @torch.no_grad()
    def update_hessian(self, module_name: str, data: torch.Tensor) -> None:
        """
        Update the Hessian state by computing covariance of the per-sample
        gradients.
        """
        hessian_state = self._state.hessian_state
        sample_counter = self._state.sample_counter

        flat_grad = rearrange(data, "b ... -> b (...)")

        # update covariance
        if module_name not in hessian_state:
            hessian_state[module_name] = torch.zeros(
                (flat_grad.shape[-1], flat_grad.shape[-1]), device="cpu"
            )
            sample_counter[module_name] = 0

        if flat_grad.is_cuda:
            # By default, the hessian state is stored on the CPU. However,
            # computing/updating the hessian state is on CPU is slow. Thus,
            # we move the hessian state to the GPU if the activation is on
            # the GPU, and then move it back to the CPU asynchrously.
            hessian_state_gpu = hessian_state[module_name].to(device=flat_grad.device)
            hessian_state_gpu.addmm_(flat_grad.t(), flat_grad)
            hessian_state[module_name] = hessian_state_gpu.to(
                device="cpu", non_blocking=True
            )
        else:
            hessian_state[module_name].addmm_(flat_grad.t(), flat_grad)
        sample_counter[module_name] += data.shape[0]
