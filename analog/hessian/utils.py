from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()
    return x


class InvalidModuleError(Exception):
    # Raised when the provided module is invalid.
    pass


def extract_patches(
    inputs: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> torch.Tensor:
    """Extract patches for the KFC approximation.

    This method is based on the technique described in https://arxiv.org/pdf/1602.01407.pdf.

    Args:
        inputs (torch.Tensor):
            Activations before the convolutional layer.
        kernel_size (tuple):
            Dimensions of the convolutional layer's kernel.
        stride (tuple):
            Stride applied in the convolutional layer.
        padding (tuple):
            Padding dimensions applied in the convolutional layer.
    """
    if padding[0] + padding[1] > 0:
        inputs = torch.nn.functional.pad(
            inputs,
            (padding[1], padding[1], padding[0], padding[0]),
        ).data
    inputs = inputs.unfold(2, kernel_size[0], stride[0])
    inputs = inputs.unfold(3, kernel_size[1], stride[1])
    inputs = inputs.transpose_(1, 2).transpose_(2, 3).contiguous()
    inputs = inputs.view(
        inputs.size(0),
        inputs.size(1),
        inputs.size(2),
        inputs.size(3) * inputs.size(4) * inputs.size(5),
    )
    return inputs


def extract_forward_activations(
    activations: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    """Extract and reshape activations into valid shapes for covariance computations.

    Args:
        activations (torch.Tensor):
            Raw pre-activations supplied to the module.
        module (nn.Module):
            The module where the activations are applied.
    """
    if isinstance(module, nn.Linear):
        reshaped_activations = activations.reshape(-1, activations.shape[-1])

    elif isinstance(module, nn.Conv2d):
        reshaped_activations = extract_patches(
            activations, module.kernel_size, module.stride, module.padding
        )
        reshaped_activations = reshaped_activations.view(
            -1, reshaped_activations.size(-1)
        )
    else:
        raise InvalidModuleError()
    return reshaped_activations


def extract_backward_activations(
    gradients: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    """Extract and reshape gradients into valid shapes for covariance computations.

    Args:
        gradients (torch.Tensor):
            Raw gradients on the output to the module.
        module (nn.Module):
            The module where the gradients are computed.
    """
    if isinstance(module, nn.Linear):
        del module
        reshaped_grads = gradients.reshape(-1, gradients.shape[-1])
        return reshaped_grads
    elif isinstance(module, nn.Conv2d):
        del module
        reshaped_grads = gradients.permute(0, 2, 3, 1)
        reshaped_grads = reshaped_grads.reshape(-1, reshaped_grads.size(-1))
    else:
        raise InvalidModuleError()
    return reshaped_grads
