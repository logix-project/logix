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

import math

import torch
import torch.nn as nn

from logix.lora.utils import compute_top_k_singular_vectors


class LoraLinear(nn.Module):
    def __init__(self, rank: int, linear: nn.Linear, shared_module: nn.Linear = None):
        """Transforms a linear layer into a LoraLinear layer.

        Args:
            rank (int): The rank of lora
            linear (nn.Linear): The linear layer to transform
        """
        super().__init__()

        in_features = linear.in_features
        out_features = linear.out_features

        self.rank = min(rank, in_features, out_features)

        self.logix_lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.logix_lora_B = shared_module or nn.Linear(self.rank, self.rank, bias=False)
        self.logix_lora_C = nn.Linear(self.rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.logix_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.logix_lora_B.weight)
        nn.init.kaiming_uniform_(self.logix_lora_C.weight, a=math.sqrt(5))

        self._linear = linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self._linear(input)
        result += self.logix_lora_C(self.logix_lora_B(self.logix_lora_A(input)))

        return result

    def pca_init_weight(self, covariance=None):
        """Initialize the weight of the LoraLinear layer.

        Args:
            init_strategy (str): The type of projection to use
            covariance (dict): The forward and backward covariance of the layer
        """
        (
            top_r_singular_vector_forward,
            top_r_singular_value_forward,
        ) = compute_top_k_singular_vectors(covariance["forward"], self.rank)
        (
            top_r_singular_vector_backward,
            top_r_singular_value_backward,
        ) = compute_top_k_singular_vectors(covariance["backward"], self.rank)
        self.logix_lora_A.weight.data.copy_(top_r_singular_vector_forward.T)
        self.logix_lora_C.weight.data.copy_(top_r_singular_vector_backward)


class LoraConv2d(nn.Module):
    def __init__(self, rank: int, conv: nn.Conv2d, shared_module: nn.Conv2d = None):
        """Transforms a conv2d layer into a LoraConv2d layer.

        Args:
            rank (int): The rank of lora
            conv (nn.Conv2d): The conv2d layer to transform
        """
        super().__init__()

        in_channels = conv.in_channels
        out_channels = conv.out_channels
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding

        self.rank = min(rank, in_channels, out_channels)

        self.logix_lora_A = nn.Conv2d(
            in_channels, self.rank, kernel_size, stride, padding, bias=False
        )
        self.logix_lora_B = shared_module or nn.Conv2d(
            self.rank, self.rank, 1, bias=False
        )
        self.logix_lora_C = nn.Conv2d(self.rank, out_channels, 1, bias=False)

        nn.init.kaiming_uniform_(self.logix_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.logix_lora_B.weight)
        nn.init.kaiming_uniform_(self.logix_lora_C.weight, a=math.sqrt(5))

        self._conv = conv

    def forward(self, input) -> torch.Tensor:
        result = self._conv(input)
        result += self.logix_lora_C(self.logix_lora_B(self.logix_lora_A(input)))

        return result

    def pca_init_weight(self, covariance):
        """Initialize the weight of the LoraConv2d layer.

        Args:
            projection_type (str): The type of projection to use
            covariance (dict): The forward and backward covariance of the layer
        """
        (
            top_r_singular_vector_forward,
            top_r_singular_value_forward,
        ) = compute_top_k_singular_vectors(covariance["forward"], self.rank)
        (
            top_r_singular_vector_backward,
            top_r_singular_value_backward,
        ) = compute_top_k_singular_vectors(covariance["backward"], self.rank)
        shape_A = self.logix_lora_A.weight.shape
        shape_C = self.logix_lora_C.weight.shape
        self.logix_lora_A.weight.data.copy_(
            top_r_singular_vector_forward.T.view(shape_A)
        )
        self.logix_lora_C.weight.data.copy_(
            top_r_singular_vector_backward.view(shape_C)
        )


class LoraEmbedding(nn.Module):
    def __init__(
        self, rank: int, embedding: nn.Embedding, shared_module: nn.Embedding = None
    ):
        """Transforms a linear layer into a LoraEmbedding layer.

        Args:
            rank (int): The rank of lora
            linear (nn.Linear): The linear layer to transform
        """
        super().__init__()

        num_embeddings = embedding.num_embeddings
        embedding_dim = embedding.embedding_dim

        self.rank = min(rank, num_embeddings, embedding_dim)

        self.logix_lora_A = nn.Embedding(num_embeddings, self.rank)
        self.logix_lora_B = shared_module or nn.Linear(self.rank, self.rank, bias=False)
        self.logix_lora_C = nn.Linear(self.rank, embedding_dim, bias=False)

        nn.init.kaiming_uniform_(self.logix_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.logix_lora_B.weight)
        nn.init.kaiming_uniform_(self.logix_lora_C.weight, a=math.sqrt(5))

        self._embedding = embedding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self._embedding(input)
        result += self.logix_lora_C(self.logix_lora_B(self.logix_lora_A(input)))

        return result

    def pca_init_weight(self, covariance=None):
        """Initialize the weight of the LoraEmbedding layer.

        Args:
            init_strategy (str): The type of projection to use
            covariance (dict): The forward and backward covariance of the layer
        """
        raise NotImplementedError
