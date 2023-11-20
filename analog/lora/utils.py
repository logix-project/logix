import torch


def compute_top_k_singular_vectors(matrix, k):
    """
    Compute the top k singular vectors of a matrix.
    """
    U, S, Vh = torch.linalg.svd(matrix)
    top_k_singular_vectors = U[:, :k]
    top_k_singular_values = S[:k]
    return top_k_singular_vectors, top_k_singular_values
