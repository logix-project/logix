import torch
import torch.nn as nn
import torch.distributed as dist

from analog.utils import deep_get, get_world_size
from analog.hessian.base import HessianHandlerBase
from analog.hessian.utils import extract_patches, try_contiguous


def get_module_type_str(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        return "conv2d"
    elif isinstance(module, nn.Linear):
        return "linear"
    else:
        raise ValueError


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
    ):
        module_type = get_module_type_str(module)
        covariance_func = getattr(self, f"{module_type}_{mode}")
        covariance = covariance_func(data, module).cpu().detach()
        if deep_get(self.hessian_state, [module_name, mode]) is None:
            self.hessian_state[module_name][mode] = covariance
        else:
            self.hessian_state[module_name][mode] += covariance

    def finalize(self):
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

    @staticmethod
    def conv2d_forward(acts, module: nn.Conv2d):
        acts = extract_patches(
            acts, module.kernel_size, module.stride, module.padding, module.groups
        )
        spatial_size = acts.size(1) * acts.size(2)
        acts = acts.view(-1, acts.size(-1))

        #! Ignore bias for now
        # if module.bias is not None:
        #     shape = list(acts.shape[:-1]) + [1]
        #     acts = torch.cat([acts, acts.new_ones(shape)], dim=-1)
        acts = acts / spatial_size
        return torch.matmul(torch.t(acts), acts)

    @staticmethod
    def linear_forward(acts, module: nn.Linear):
        acts = acts.reshape(-1, acts.size(-1))

        #! Ignore bias for now
        # if module.bias is not None:
        #     shape = list(acts.shape[:-1]) + [1]
        #     acts = torch.cat([acts, acts.new_ones(shape)], dim=-1)
        return torch.matmul(torch.t(acts), acts)

    @staticmethod
    def conv2d_backward(grads, module: nn.Conv2d):
        del module
        spatial_size = grads.size(2) * grads.size(3)
        grads = grads.transpose(1, 2).transpose(2, 3)
        grads = grads.permute(1, 0, 2, 3)
        grads = try_contiguous(grads)
        grads = grads.view(-1, grads.size(-1))
        grads = grads / spatial_size
        return torch.matmul(torch.t(grads), grads)

    @staticmethod
    def linear_backward(grads, module: nn.Linear):
        del module
        grads = grads.reshape(-1, grads.size(-1))
        return torch.matmul(grads.t(), grads)
