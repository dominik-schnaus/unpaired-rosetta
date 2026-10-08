"""Preprocessing of embeddings and the polar factor used by orthogonal Procrustes."""

import torch
from torch import Tensor

NORM_FLOOR = 1e-10


def center_and_normalize(
    samples: Tensor,
    mean: Tensor,
) -> Tensor:
    """Subtract the training mean of the space and scale every row to unit length."""
    centered = samples - mean
    return centered / centered.norm(dim=-1, keepdim=True).clamp(min=NORM_FLOOR)


def polar(
    matrix: Tensor,
) -> Tensor:
    """Semi-orthogonal polar factor ``U V^T`` of ``matrix = U S V^T``.

    For row-aligned ``X`` and ``Y``, ``polar(X^T Y) = argmax_{W in St(d_X, d_Y)} Tr(X W Y^T)`` (orthogonal Procrustes).
    """
    try:
        left, _, right_transposed = torch.linalg.svd(matrix, full_matrices=False)
    except torch.linalg.LinAlgError:  # the float32 SVD can fail to converge, float64 does not
        left, _, right_transposed = torch.linalg.svd(matrix.double(), full_matrices=False)
        left, right_transposed = left.to(matrix.dtype), right_transposed.to(matrix.dtype)
    return left @ right_transposed
