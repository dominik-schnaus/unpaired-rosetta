"""Local CKA (Maniparambil et al., 2024), which scores a query and a key by how well they extend the known pairs.

The score of a query ``x`` and a key ``y`` is ``CKA(vstack(X_hat, x), vstack(Y_hat, y))`` for the known pairs ``(X_hat, Y_hat)``.
The official code computes one CKA per candidate pair. We compute the same value for all pairs at once from a
closed-form expansion of the HSIC of ``p + 1`` samples. As in Sec. 4.3, each feature is first divided by its
standard deviation on the unpaired training set. Without known pairs the similarity is constant (chance level).
"""

import torch
from torch import Tensor

from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.randomness import Randomness

FLOOR = 1e-10  # lower bound for the standard deviations and the CKA denominator


def local_hsic(
    base_x: Tensor,
    base_y: Tensor,
    query_x: Tensor,
    query_y: Tensor,
    cross_x: Tensor,
    cross_y: Tensor,
) -> Tensor:
    """``[M, L]`` unnormalized HSIC of the ``p`` base pairs extended by query ``i`` of X and key ``j`` of Y.

    Args:
        base_x, base_y: ``[p, p]`` linear kernels of the base sets.
        query_x, query_y: squared norms of the ``M`` queries and ``L`` keys.
        cross_x, cross_y: ``[p, M]`` and ``[p, L]`` kernels between the base set and the queries or keys.
    """
    n = base_x.shape[0]
    centered_x = base_x - base_x.sum(dim=1, keepdim=True) / (n + 1)
    centered_y = base_y - base_y.sum(dim=1, keepdim=True) / (n + 1)
    return (
        (centered_x * centered_y.T).sum()
        + 2 * (n / (n + 1)) * cross_x.T @ cross_y
        - 2 * ((n - 1) / (n + 1) ** 2) * torch.outer(cross_x.sum(dim=0), cross_y.sum(dim=0))
        - (2 / (n + 1)) * (cross_x.T @ centered_y.sum(dim=0)[:, None] + centered_x.sum(dim=0)[None, :] @ cross_y)
        + (n / (n + 1)) ** 2 * torch.outer(query_x, query_y)
        - 2 * (n / (n + 1) ** 2) * (torch.outer(cross_x.sum(dim=0), query_y) + torch.outer(query_x, cross_y.sum(dim=0)))
        + (1 / (n + 1) ** 2) * (base_y.sum() * query_x[:, None] + base_x.sum() * query_y[None, :])
    )


def local_hsic_self(
    base: Tensor,
    query: Tensor,
    cross: Tensor,
) -> Tensor:
    """``local_hsic`` of a space with itself, only for each query paired with itself (a vector)."""
    n = base.shape[0]
    centered = base - base.sum(dim=1, keepdim=True) / (n + 1)
    return (
        (centered * centered.T).sum()
        + 2 * (n / (n + 1)) * (cross * cross).sum(dim=0)
        - 2 * ((n - 1) / (n + 1) ** 2) * cross.sum(dim=0) ** 2
        - (2 / (n + 1)) * 2 * cross.T @ centered.sum(dim=0)
        + (n / (n + 1)) ** 2 * query**2
        - 2 * (n / (n + 1) ** 2) * 2 * cross.sum(dim=0) * query
        + (1 / (n + 1) ** 2) * 2 * base.sum() * query
    )


class LocalCKA:
    def __init__(
        self,
    ) -> None:
        self.std_x = self.std_y = self.base_x = self.base_y = None

    @torch.inference_mode()
    @torch_threads()
    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None = None,
        paired_y: Tensor | None = None,
        randomness: Randomness | None = None,
    ) -> "LocalCKA":
        self.std_x = samples_x.std(dim=0, keepdim=True) + FLOOR
        self.std_y = samples_y.std(dim=0, keepdim=True) + FLOOR
        if paired_x is not None and paired_x.shape[0] > 0:
            self.base_x = paired_x / self.std_x
            self.base_y = paired_y / self.std_y
        return self

    @torch.inference_mode()
    def similarity(
        self,
        queries_x: Tensor,
        keys_y: Tensor,
    ) -> Tensor:
        if self.base_x is None:
            return torch.zeros(queries_x.shape[0], keys_y.shape[0])
        queries_x, keys_y = queries_x / self.std_x, keys_y / self.std_y
        kernel_x, kernel_y = self.base_x @ self.base_x.T, self.base_y @ self.base_y.T
        norms_x, norms_y = (queries_x**2).sum(dim=1), (keys_y**2).sum(dim=1)
        cross_x, cross_y = self.base_x @ queries_x.T, self.base_y @ keys_y.T
        hsic = local_hsic(kernel_x, kernel_y, norms_x, norms_y, cross_x, cross_y)
        hsic_x = local_hsic_self(kernel_x, norms_x, cross_x)
        hsic_y = local_hsic_self(kernel_y, norms_y, cross_y)
        # rounding can make a self-HSIC slightly negative
        return hsic / (hsic_x[:, None] * hsic_y[None, :]).clamp(min=0).sqrt().clamp(min=FLOOR)
