from typing import List

import torch


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
