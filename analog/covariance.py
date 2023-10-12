import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovarianceHandler:
    def compute(self, data):
        """Compute the covariance for given data."""
        raise NotImplementedError


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def extract_patches(x, kernel_size, stride, padding, groups):
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0])).data
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    return torch.mean(
        x.reshape((x.size(0), x.size(1), x.size(2), groups, -1, x.size(4), x.size(5))),
        3,
    ).view(x.size(0), x.size(1), x.size(2), -1)


class GradientsHandler:
    @classmethod
    def __call__(cls, inputs, grad_outputs, layer):
        with torch.no_grad():
            if isinstance(layer, nn.Linear):
                grads = cls.linear(inputs, grad_outputs, layer)
            elif isinstance(layer, nn.Conv2d):
                grads = cls.conv2d(inputs, grad_outputs, layer)
            else:
                raise NotImplementedError()
            return grads

    @staticmethod
    def linear(inputs, grad_outputs, layer):
        if layer.bias is not None:
            inputs = torch.cat([inputs, inputs.new(inputs.size(0), 1).fill_(1)], 1)
        inputs = inputs.unsqueeze(1)
        grad_outputs = grad_outputs.unsqueeze(2)
        grads = torch.bmm(grad_outputs, inputs)
        return grads

    @staticmethod
    def conv2d(inputs, grad_outputs, layer):
        inputs = extract_patches(
            inputs, layer.kernel_size, layer.stride, layer.padding, layer.groups
        )
        inputs = inputs.view(-1, inputs.size(-1))
        grad_outputs = grad_outputs.transpose(1, 2).transpose(2, 3)
        grad_outputs = try_contiguous(grad_outputs).view(
            grad_outputs.size(0), -1, grad_outputs.size(-1)
        )
        if layer.bias is not None:
            inputs = torch.cat([inputs, inputs.new(inputs.size(0), 1).fill_(1)], 1)
        inputs = inputs.view(grad_outputs.size(0), -1, inputs.size(-1))
        grads = torch.einsum("abm,abn->amn", (grad_outputs, inputs))
        return grads


class ActivationHandler:
    """Save activation instead of its covariance"""

    @classmethod
    def compute_cov_a(cls, acts, module):
        return cls.__call__(acts, module)

    @classmethod
    def __call__(cls, acts, module):
        with torch.no_grad():
            if isinstance(module, nn.Linear):
                cov_a = cls.linear(acts, module)
            elif isinstance(module, nn.Conv2d):
                cov_a = cls.conv2d(acts, module)
            else:
                raise NotImplementedError()
            return cov_a

    @staticmethod
    def conv2d(acts, module):
        acts = extract_patches(
            acts, module.kernel_size, module.stride, module.padding, module.groups
        )
        spatial_size = acts.size(1) * acts.size(2)
        acts = acts.view(-1, acts.size(-1))
        if module.bias is not None:
            shape = list(acts.shape[:-1]) + [1]
            acts = torch.cat([acts, acts.new_ones(shape)], dim=-1)
        acts = acts / spatial_size
        return acts

    @staticmethod
    def linear(acts, layer):
        if layer.bias is not None:
            shape = list(acts.shape[:-1]) + [1]
            acts = torch.cat([acts, acts.new_ones(shape)], dim=-1)
        return acts


class PseudoGradientHandler:
    @classmethod
    def compute_cov_g(cls, g, layer):
        return cls.__call__(g, layer)

    @classmethod
    def __call__(cls, g, layer):
        with torch.no_grad():
            if isinstance(layer, nn.Conv2d):
                cov_g = cls.conv2d(g, layer)
            elif isinstance(layer, nn.Linear):
                cov_g = cls.linear(g, layer)
            else:
                raise NotImplementedError()
            return cov_g

    @staticmethod
    def conv2d(grads, layer):
        del layer
        spatial_size = grads.size(2) * grads.size(3)
        grads = grads.transpose(1, 2).transpose(2, 3)
        grads = try_contiguous(grads)
        grads = grads.view(-1, grads.size(-1))
        grads = grads / spatial_size
        return grads

    @staticmethod
    def linear(grads, layer):
        del layer
        return grads
