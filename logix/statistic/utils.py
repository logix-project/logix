# Copyright 2023-present the LogIX team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvalidModuleError(Exception):
    """
    Raised when the provided module is invalid.
    """

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
        inputs = F.pad(inputs, (padding[1], padding[1], padding[0], padding[0])).data
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


def make_forward_2d(
    data: torch.Tensor,
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
        reshaped_data = data.reshape(-1, data.shape[-1])
    elif isinstance(module, nn.Conv2d):
        reshaped_data = extract_patches(
            data, module.kernel_size, module.stride, module.padding
        )
        reshaped_data = reshaped_data.view(-1, reshaped_data.size(-1))
    else:
        raise InvalidModuleError()
    return reshaped_data


def make_backward_2d(
    data: torch.Tensor,
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
        reshaped_data = data.reshape(-1, data.shape[-1])
    elif isinstance(module, nn.Conv2d):
        del module
        reshaped_data = data.permute(0, 2, 3, 1)
        reshaped_data = reshaped_data.reshape(-1, reshaped_data.size(-1))
    else:
        raise InvalidModuleError()
    return reshaped_data


def make_2d(data: torch.Tensor, module: nn.Module, log_type: str) -> torch.Tensor:
    """Extract and reshape data into 2d for computing statistic.

    Args:
        module (nn.Module):
            The module where the activations are applied.
        mode (str):
            Forward or backward.
        data (torch.Tensor):
            Activations corresponding to the module.
    """
    if module is None:
        return data.reshape(data.shape[0], -1)
    elif log_type == "forward":
        return make_forward_2d(data, module)
    elif log_type == "backward":
        return make_backward_2d(data, module)
    else:
        raise ValueError(f"Invalid mode {log_type}")
