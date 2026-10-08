"""Scores of how similar the geometries of two row-aligned embedding sets are.

* CKA: linear centered kernel alignment (Kornblith et al., 2019) with the unbiased HSIC estimator.
* Mutual k-NN: mean overlap of the k nearest neighbors in both spaces (Huh et al., 2024).
* TSI and QSI: fraction of sampled triplets or quadruplets whose distance order agrees in both spaces
  (Soares et al., 2026).
* Clipped CKA (App. D): the CKA variant in the code that Wang et al. (2026) reuse, on clipped and normalized features.

The ``*_of_kernels`` variants take two similarity kernels instead of embeddings (App. E). Clipped kernels have exact
ties. Order scores count a tie as half an agreement, and nearest neighbors break ties at random.
"""

import torch
from torch import Tensor
from torch.nn.functional import normalize

CLIPPING_QUANTILE = 0.95
TIE_JITTER = 1e-12  # smaller than any gap between distinct similarities, so it only breaks exact ties


def cka(
    samples_x: Tensor,
    samples_y: Tensor,
    block_size: int = 4096,
) -> float:
    """Unbiased linear CKA ``HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L))`` for ``K = X X^T`` and ``L = Y Y^T``.

    Computed in blocks of rows, so no ``n x n`` kernel is stored. The unbiased HSIC (Song et al., 2012) of kernels
    ``K0``, ``L0`` with zero diagonals is ``[Tr(K0 L0) + 1^T K0 1 1^T L0 1 / ((n-1)(n-2)) - 2 1^T K0 L0 1 / (n-2)]
    / (n (n-3))``.
    """
    n = samples_x.shape[0]
    if n <= 3:
        raise ValueError(f"The unbiased CKA needs at least 4 samples, got {n}.")
    x, y = samples_x.double(), samples_y.double()
    trace_kl = trace_kk = trace_ll = 0.0
    rows_k, rows_l, diagonal_k, diagonal_l = [], [], [], []
    for start in range(0, n, block_size):
        kernel_x, kernel_y = x[start : start + block_size] @ x.T, y[start : start + block_size] @ y.T
        trace_kl += (kernel_x * kernel_y).sum().item()
        trace_kk += (kernel_x * kernel_x).sum().item()
        trace_ll += (kernel_y * kernel_y).sum().item()
        rows_k.append(kernel_x.sum(dim=1))
        rows_l.append(kernel_y.sum(dim=1))
        diagonal_k.append(kernel_x.diagonal(offset=start))
        diagonal_l.append(kernel_y.diagonal(offset=start))
    rows_k, rows_l = torch.cat(rows_k), torch.cat(rows_l)
    diagonal_k, diagonal_l = torch.cat(diagonal_k), torch.cat(diagonal_l)

    def hsic(
        trace: float,
        rows1: Tensor,
        rows2: Tensor,
        diagonal1: Tensor,
        diagonal2: Tensor,
    ) -> float:
        """Unbiased HSIC from the traces, row sums and diagonals of the two full kernels."""
        trace_without_diagonal = trace - (diagonal1 * diagonal2).sum().item()
        sum1 = (rows1.sum() - diagonal1.sum()).item()
        sum2 = (rows2.sum() - diagonal2.sum()).item()
        row_products = (
            (rows1 * rows2).sum()
            - (rows1 * diagonal2).sum()
            - (rows2 * diagonal1).sum()
            + (diagonal1 * diagonal2).sum()
        ).item()
        return (trace_without_diagonal + sum1 * sum2 / ((n - 1) * (n - 2)) - 2 * row_products / (n - 2)) / (n * (n - 3))

    hsic_kl = hsic(trace_kl, rows_k, rows_l, diagonal_k, diagonal_l)
    hsic_kk = hsic(trace_kk, rows_k, rows_k, diagonal_k, diagonal_k)
    hsic_ll = hsic(trace_ll, rows_l, rows_l, diagonal_l, diagonal_l)
    return hsic_kl / (hsic_kk**0.5 * hsic_ll**0.5)


def nearest_neighbors(
    samples: Tensor,
    num_neighbors: int,
    block_size: int = 4096,
) -> Tensor:
    """Indices of the ``k`` largest inner products of each sample with the other samples."""
    n = samples.shape[0]
    neighbors = torch.empty(n, num_neighbors, dtype=torch.long)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        candidates = (samples[start:stop] @ samples.T).topk(num_neighbors + 1, dim=1).indices
        is_self = candidates == torch.arange(start, stop)[:, None]
        # Drop the sample itself, or the last candidate if the sample is not among them.
        drop = torch.where(is_self.any(dim=1), is_self.long().argmax(dim=1), num_neighbors)
        keep = torch.ones_like(candidates, dtype=torch.bool)
        keep[torch.arange(stop - start), drop] = False
        neighbors[start:stop] = candidates[keep].view(stop - start, num_neighbors)
    return neighbors


def neighbor_overlap(
    neighbors_x: Tensor,
    neighbors_y: Tensor,
) -> float:
    """Mean fraction of shared entries of two neighbor tables ``[n, k]``."""
    shared = (neighbors_x[:, :, None] == neighbors_y[:, None, :]).any(dim=2).sum(dim=1)
    return (shared.float() / neighbors_x.shape[1]).mean().item()


def mutual_knn(
    samples_x: Tensor,
    samples_y: Tensor,
    num_neighbors: int = 10,
) -> float:
    """Mean fraction of shared k nearest neighbors (by inner product) of each sample in the two spaces."""
    return neighbor_overlap(nearest_neighbors(samples_x, num_neighbors), nearest_neighbors(samples_y, num_neighbors))


def triplet_similarity_index(
    samples_x: Tensor,
    samples_y: Tensor,
    generator: torch.Generator,
    num_samples: int = 100_000,
) -> float:
    """Fraction of random triplets whose order ``d(i, j) < d(i, k)`` agrees in both spaces."""
    n = samples_x.shape[0]
    i, j, k = (torch.randint(0, n, (num_samples,), generator=generator) for _ in range(3))
    distinct = (i != j) & (i != k) & (j != k)
    i, j, k = i[distinct], j[distinct], k[distinct]
    closer_x = (samples_x[i] - samples_x[j]).norm(dim=1) < (samples_x[i] - samples_x[k]).norm(dim=1)
    closer_y = (samples_y[i] - samples_y[j]).norm(dim=1) < (samples_y[i] - samples_y[k]).norm(dim=1)
    return (closer_x == closer_y).float().mean().item()


def quadruplet_similarity_index(
    samples_x: Tensor,
    samples_y: Tensor,
    generator: torch.Generator,
    num_samples: int = 100_000,
) -> float:
    """Fraction of random quadruplets whose order ``d(i, j) < d(k, l)`` agrees in both spaces."""
    n = samples_x.shape[0]
    i, j, k, l = (torch.randint(0, n, (num_samples,), generator=generator) for _ in range(4))  # noqa: E741
    valid = (i != j) & (k != l)
    i, j, k, l = i[valid], j[valid], k[valid], l[valid]  # noqa: E741
    closer_x = (samples_x[i] - samples_x[j]).norm(dim=1) < (samples_x[k] - samples_x[l]).norm(dim=1)
    closer_y = (samples_y[i] - samples_y[j]).norm(dim=1) < (samples_y[k] - samples_y[l]).norm(dim=1)
    return (closer_x == closer_y).float().mean().item()


def geometry_scores(
    samples_x: Tensor,
    samples_y: Tensor,
    generator: torch.Generator,
) -> dict[str, float]:
    """The four scores of the paper. The order fixes the random draws."""
    return {
        "cka": cka(samples_x, samples_y),
        "mutual_knn": mutual_knn(samples_x, samples_y, num_neighbors=10),
        "tsi": triplet_similarity_index(samples_x, samples_y, generator),
        "qsi": quadruplet_similarity_index(samples_x, samples_y, generator),
    }


def clip_outliers(
    samples: Tensor,
) -> Tensor:
    """Clamp entries to ``+-b``, where ``b`` is the mean over samples of the 95th percentile of absolute entries.

    Matches ``remove_outliers(exact=False)`` of the platonic-representation code.
    """
    bound = torch.quantile(samples.abs(), CLIPPING_QUANTILE, dim=1).mean()
    return samples.clamp(-bound, bound)


def clipped_cka(
    samples_x: Tensor,
    samples_y: Tensor,
) -> float:
    """Unbiased CKA of clipped, normalized and uncentered features, as in Wang et al. (2026).

    Unlike their code, we add no 1e-6 to the denominator and use float64.
    """
    unit_x = normalize(clip_outliers(samples_x.double()), dim=-1)
    unit_y = normalize(clip_outliers(samples_y.double()), dim=-1)
    return cka(unit_x, unit_y)


def cka_of_kernels(
    kernel_x: Tensor,
    kernel_y: Tensor,
) -> float:
    """Unbiased CKA of two full ``n x n`` kernels."""
    n = kernel_x.shape[0]
    zero_diagonal_x = kernel_x.double().clone().fill_diagonal_(0)
    zero_diagonal_y = kernel_y.double().clone().fill_diagonal_(0)

    def hsic(
        first: Tensor,
        second: Tensor,
    ) -> Tensor:
        return (
            (first * second).sum()
            + first.sum() * second.sum() / ((n - 1) * (n - 2))
            - 2 * (first.sum(0) @ second.sum(1)) / (n - 2)
        ) / (n * (n - 3))

    hsic_xy = hsic(zero_diagonal_x, zero_diagonal_y)
    return (hsic_xy / (hsic(zero_diagonal_x, zero_diagonal_x) * hsic(zero_diagonal_y, zero_diagonal_y)).sqrt()).item()


def kernel_neighbors(
    kernel: Tensor,
    num_neighbors: int,
    generator: torch.Generator,
) -> Tensor:
    """Indices of the ``k`` largest off-diagonal similarities in each row. Exact ties are broken at random."""
    ranked = kernel.double() + TIE_JITTER * torch.rand(kernel.shape, generator=generator, dtype=torch.float64)
    ranked.fill_diagonal_(float("-inf"))
    return ranked.topk(num_neighbors, dim=1).indices


def ordering_agreement(
    first_x: Tensor,
    second_x: Tensor,
    first_y: Tensor,
    second_y: Tensor,
) -> float:
    """Fraction of pairs ordered the same way in both spaces. A tie in either space counts half."""
    sign_x, sign_y = torch.sign(first_x - second_x), torch.sign(first_y - second_y)
    tied = (sign_x == 0) | (sign_y == 0)
    return torch.where(tied, 0.5, (sign_x == sign_y).double()).mean().item()


def triplet_similarity_index_of_kernels(
    kernel_x: Tensor,
    kernel_y: Tensor,
    generator: torch.Generator,
    num_samples: int = 100_000,
) -> float:
    """TSI on similarity kernels, comparing ``s(i, j)`` with ``s(i, k)``."""
    n = kernel_x.shape[0]
    i, j, k = (torch.randint(0, n, (num_samples,), generator=generator) for _ in range(3))
    distinct = (i != j) & (i != k) & (j != k)
    i, j, k = i[distinct], j[distinct], k[distinct]
    return ordering_agreement(kernel_x[i, j], kernel_x[i, k], kernel_y[i, j], kernel_y[i, k])


def quadruplet_similarity_index_of_kernels(
    kernel_x: Tensor,
    kernel_y: Tensor,
    generator: torch.Generator,
    num_samples: int = 100_000,
) -> float:
    """QSI on similarity kernels, comparing ``s(i, j)`` with ``s(k, l)``."""
    n = kernel_x.shape[0]
    i, j, k, l = (torch.randint(0, n, (num_samples,), generator=generator) for _ in range(4))  # noqa: E741
    valid = (i != j) & (k != l)
    i, j, k, l = i[valid], j[valid], k[valid], l[valid]  # noqa: E741
    return ordering_agreement(kernel_x[i, j], kernel_x[k, l], kernel_y[i, j], kernel_y[k, l])
