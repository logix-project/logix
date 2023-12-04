import math

import torch
import torch.nn as nn

from analog.constants import FORWARD, BACKWARD
from analog.lora.utils import compute_top_k_singular_vectors


class LoraLinear(nn.Linear):
    def __init__(self, rank: int, linear: nn.Linear, shared_module: nn.Linear = None):
        """Transforms a linear layer into a LoraLinear layer.

        Args:
            rank (int): The rank of lora
            linear (nn.Linear): The linear layer to transform
        """
        in_features = linear.in_features
        out_features = linear.out_features

        super().__init__(in_features, out_features)
        self.rank = min(rank, in_features, out_features)

        self.analog_lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.analog_lora_B = shared_module or nn.Linear(
            self.rank, self.rank, bias=False
        )
        self.analog_lora_C = nn.Linear(self.rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.analog_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.analog_lora_B.weight)
        nn.init.kaiming_uniform_(self.analog_lora_C.weight, a=math.sqrt(5))

        self._linear = linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self._linear(input)
        result += self.analog_lora_C(self.analog_lora_B(self.analog_lora_A(input)))

        return result

    def pca_init_weight(self, init_strategy: str = "random", hessian=None):
        """Initialize the weight of the LoraLinear layer.

        Args:
            init_strategy (str): The type of projection to use
            hessian (dict): The forward and backward hessian of the layer
        """
        (
            top_r_singular_vector_forward,
            top_r_singular_value_forward,
        ) = compute_top_k_singular_vectors(hessian[FORWARD], self.rank)
        (
            top_r_singular_vector_backward,
            top_r_singular_value_backward,
        ) = compute_top_k_singular_vectors(hessian[BACKWARD], self.rank)
        self.analog_lora_A.weight.data.copy_(top_r_singular_vector_forward.T)
        self.analog_lora_C.weight.data.copy_(top_r_singular_vector_backward)


class LoraConv2d(nn.Conv2d):
    def __init__(self, rank: int, conv: nn.Conv2d, shared_module: nn.Conv2d = None):
        """Transforms a conv2d layer into a LoraConv2d layer.

        Args:
            rank (int): The rank of lora
            conv (nn.Conv2d): The conv2d layer to transform
        """
        in_channels = conv.in_channels
        out_channels = conv.out_channels
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding

        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )

        self.rank = min(rank, self.in_channels, self.out_channels)

        self.analog_lora_A = nn.Conv2d(
            self.in_channels, self.rank, kernel_size, stride, padding, bias=False
        )
        self.analog_lora_B = shared_module or nn.Conv2d(
            self.rank, self.rank, 1, bias=False
        )
        self.analog_lora_C = nn.Conv2d(self.rank, self.out_channels, 1, bias=False)

        nn.init.kaiming_uniform_(self.analog_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.analog_lora_B.weight)
        nn.init.kaiming_uniform_(self.analog_lora_C.weight, a=math.sqrt(5))

        self._conv = conv

    def forward(self, input) -> torch.Tensor:
        result = self._conv(input)
        result += self.analog_lora_C(self.analog_lora_B(self.analog_lora_A(input)))

        return result

    def pca_init_weight(self, projection_type, hessian):
        """Initialize the weight of the LoraLinear layer.

        Args:
            projection_type (str): The type of projection to use
            hessian (dict): The forward and backward hessian of the layer
        """
        (
            top_r_singular_vector_forward,
            top_r_singular_value_forward,
        ) = compute_top_k_singular_vectors(hessian[FORWARD], self.rank)
        (
            top_r_singular_vector_backward,
            top_r_singular_value_backward,
        ) = compute_top_k_singular_vectors(hessian[BACKWARD], self.rank)
        shape_A = self.analog_lora_A.weight.shape
        shape_C = self.analog_lora_C.weight.shape
        self.analog_lora_A.weight.data.copy_(
            top_r_singular_vector_forward.T.view(shape_A)
        )
        self.analog_lora_C.weight.data.copy_(
            top_r_singular_vector_backward.view(shape_C)
        )
