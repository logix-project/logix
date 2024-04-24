from typing import List

import math
import torch
import torch.nn as nn


def find_rank_pca_covariance(matrix, threshold):
    """
    Calculate the minimum principal component analysis (PCA) rank required
    to explain at least the specified percentage (threshold) of the total covariance.
    """
    U, S, Vh = torch.linalg.svd(matrix)
    rank = 0
    cur, total = 0, sum(S)
    while rank < len(S) and (cur / total) < threshold:
        cur += S[rank]
        rank += 1

    return rank


def find_rank_pca_compression(module, ratio):
    """
    Calculate the minimum principal component analysis (PCA) rank required
    to reach threshold compression ratio.
    """
    weight = module.weight.detach().cpu().numpy()
    if isinstance(module, nn.Linear):
        # r * r = m * n * ratio
        in_features, out_features = weight.shape
        rank = math.ceil(math.sqrt(in_features * out_features * ratio))
    elif isinstance(module, nn.Conv2d):
        # r * r * 1 * 1 = in_channels * out_channels * kernel_size[0] * kernel_size[1] * ratio
        in_channels, out_channels, kernel_size0, kernel_size1 = weight.shape
        rank = math.ceil(
            math.sqrt(in_channels * out_channels * kernel_size0 * kernel_size1 * ratio)
        )
        return rank
    elif isinstance(module, nn.Embedding):
        # r * r = m * n * ratio
        num_embeddings, embedding_dim = weight.shape
        rank = math.ceil(math.sqrt(num_embeddings * embedding_dim * ratio))
    else:
        raise NotImplementedError

    return rank


def pca_rank_by_weight_shape(shape, module):
    if isinstance(module, nn.Linear):
        assert len(shape) == 2
        return shape[1], shape[0]
    elif isinstance(module, nn.Conv2d):
        assert len(shape) == 4
        return shape[1], shape[0]
    elif isinstance(module, nn.Embedding):
        assert len(shape) == 2
        return shape[1], shape[0]


def is_lora(model):
    """
    Check if LoRA is enabled for the model.
    """
    return any("logix_lora_B" in name for name, module in model.named_modules())


def compute_top_k_singular_vectors(matrix, k):
    """
    Compute the top k singular vectors of a matrix.
    """
    U, S, _ = torch.linalg.svd(matrix)
    top_k_singular_vectors = U[:, :k]
    top_k_singular_values = S[:k]
    return top_k_singular_vectors, top_k_singular_values


def find_parameter_sharing_group(
    module_name: str, parameter_sharing_groups: List[str] = None
):
    if parameter_sharing_groups is None:
        return "logix_lora_none"

    found_groups = [psg for psg in parameter_sharing_groups if psg in module_name]
    assert (
        len(found_groups) == 1
    ), "Each module can belong to only one parameter sharing group."
    return found_groups[0]


def _get_submodules(model, key):
    """
    Helper function to replace a module with transformers model
    https://github.com/huggingface/peft/blob/c0dd27bc974e4a62c6072142146887b75bb2de6c/src/peft/utils/other.py#L251
    """
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name
