import torch


def reconstruct_grad(log):
    """
    Reconstructs the gradient from forward & backward activations via outer product.

    Args:
        log (dict): The activation log.
    Returns:
        torch.Tensor: The batch of gradient.
    """
    fwd = log["forward"]
    bwd = log["backward"]
    return torch.bmm(bwd.unsqueeze(2), fwd.unsqueeze(1))


def do_decompose(src_log, tgt_log):
    return True


def rescaled_dot_product(src, tgt, scale):
    return
