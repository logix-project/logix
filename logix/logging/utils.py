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

import torch
import torch.nn as nn
from einops import einsum, reduce


@torch.no_grad()
def compute_per_sample_gradient(
    fwd: torch.Tensor, bwd: torch.Tensor, module: nn.Module
):
    """
    Computes the per-sample gradient of a module.

    Args:
        fwd: The forward activations of the module.
        bwd: The backward activations of the module.
        module: The module whose per-sample gradient needs to be computed.
    """
    if isinstance(module, nn.Linear):
        # For linear layers, we can simply compute the outer product of the
        # forward and backward activations.
        outer_product = einsum(bwd, fwd, "... i, ... j -> ... i j")
        return reduce(outer_product, "n ... i j -> n i j", "sum")
    elif isinstance(module, nn.Conv2d):
        # For convolutional layers, we need to unfold the forward activations
        # and compute the outer product of the backward and unfolded forward
        # activations.
        bsz = fwd.shape[0]
        fwd_unfold = torch.nn.functional.unfold(
            fwd,
            module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
        fwd_unfold = fwd_unfold.reshape(bsz, fwd_unfold.shape[1], -1)
        bwd = bwd.reshape(bsz, -1, fwd_unfold.shape[-1])
        grad = einsum(bwd, fwd_unfold, "i j k, i l k -> i j l")

        # Ensure that each gradient has two dimensions of (out_dim, in_dim)
        shape = [bsz, module.weight.shape[0], -1]
        return grad.reshape(shape)
    elif isinstance(module, nn.Conv1d):
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
