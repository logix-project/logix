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

        nn.init.zeros_(self.analog_lora_B.weight)

        self._linear = linear

    def forward(self, input) -> torch.Tensor:
        result = self._linear(input)
        result += self.analog_lora_C(self.analog_lora_B(self.analog_lora_A(input)))

        return result

    def init_weight(self, init_strategy: str = "random", hessian=None):
        """Initialize the weight of the LoraLinear layer.

        Args:
            init_strategy (str): The type of projection to use
            hessian (dict): The forward and backward hessian of the layer
        """
        if init_strategy == "random":
            nn.init.kaiming_uniform_(self.analog_lora_A.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.analog_lora_C.weight, a=math.sqrt(5))
        elif init_strategy == "pca":
            top_r_singular_vector_forward = compute_top_k_singular_vectors(
                hessian[FORWARD], self.rank
            )
            top_r_singular_vector_backward = compute_top_k_singular_vectors(
                hessian[BACKWARD], self.rank
            )
            self.analog_lora_A.weight.data.copy_(top_r_singular_vector_forward.T)
            self.analog_lora_C.weight.data.copy_(top_r_singular_vector_backward)
