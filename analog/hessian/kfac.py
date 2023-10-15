import torch
import torch.nn as nn

from analog.utils import deep_get
from analog.hessian.base import HessianHandlerBase
from analog.hessian.utils import extract_patches, try_contiguous


def get_module_type_str(module):
    if isinstance(module, nn.Conv2d):
        return "conv2d"
    elif isinstance(module, nn.Linear):
        return "linear"
    else:
        raise NotImplementedError


class KFACHessianHandler(HessianHandlerBase):
    def update_hessian(self, module, module_name, mode, data):
        module_type = get_module_type_str(module)
        covariance_func = getattr(self, f"{module_type}_{mode}")
        covariance = covariance_func(data, module)
        if deep_get(self.hessian_state, [module_name, mode]) is None:
            self.hessian_state[module_name][mode] = covariance.cpu()
        else:
            self.hessian_state[module_name][mode] += covariance.cpu()

    def hessian_inverse(self):
        for _, module_state in self.hessian_state.items():
            for mode, covariance in module_state.items():
                module_state[mode] = torch.inverse(
                    covariance + torch.trace(covariance) * self.config.damping
                )

    @staticmethod
    def conv2d_forward(acts, module):
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
    def linear_forward(acts, module):
        acts = acts.reshape(-1, acts.size(-1))

        #! Ignore bias for now
        # if module.bias is not None:
        #     shape = list(acts.shape[:-1]) + [1]
        #     acts = torch.cat([acts, acts.new_ones(shape)], dim=-1)
        return torch.matmul(torch.t(acts), acts)

    @staticmethod
    def conv2d_backward(grads, module):
        del module
        spatial_size = grads.size(2) * grads.size(3)
        grads = grads.transpose(1, 2).transpose(2, 3)
        grads = grads.permute(1, 0, 2, 3)
        grads = try_contiguous(grads)
        grads = grads.view(-1, grads.size(-1))
        grads = grads / spatial_size
        return torch.matmul(torch.t(grads), grads)

    @staticmethod
    def linear_backward(grads, module):
        del module
        return torch.matmul(grads.t(), grads)
