import math
import torch
import torch.nn as nn

class LoraLinear(nn.Linear):
    def __init__(self, r: int, linear: nn.Linear):
        """Transforms a linear layer into a LoraLinear layer.

        Args:
            r (int): The rank of lora
            linear (nn.Linear): The linear layer to transform
        """
        in_features = linear.in_features
        out_features = linear.out_features

        super().__init__(in_features, out_features)

        self.analog_lora_A = nn.Linear(in_features, r, bias=False)
        self.analog_lora_B = nn.Linear(r, r, bias=False)
        self.analog_lora_C = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.analog_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.analog_lora_B.weight)
        nn.init.kaiming_uniform_(self.analog_lora_C.weight, a=math.sqrt(5))

        self._linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        result = self._linear(x)
        result += self.analog_lora_C(self.analog_lora_B(self.analog_lora_A(x)))

        return result
    