"""Tests of the clipped CKA and the kernel scores against their definitions and the embedding scores."""

import pytest
import torch
from torch.nn.functional import normalize

from unpaired_rosetta.geometry import (
    cka,
    cka_of_kernels,
    clip_outliers,
    clipped_cka,
    kernel_neighbors,
    mutual_knn,
    nearest_neighbors,
    neighbor_overlap,
    ordering_agreement,
    quadruplet_similarity_index,
    quadruplet_similarity_index_of_kernels,
    triplet_similarity_index,
    triplet_similarity_index_of_kernels,
)


def paired_samples(
    num_samples: int = 200,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    samples_x = torch.randn(num_samples, 12, dtype=torch.float64, generator=generator)
    samples_y = samples_x @ torch.randn(12, 9, dtype=torch.float64, generator=generator) + torch.randn(
        num_samples, 9, dtype=torch.float64, generator=generator
    )
    return samples_x, samples_y


def test_clip_outliers_clamps_at_the_mean_per_sample_quantile() -> None:
    samples = torch.randn(50, 40, dtype=torch.float64) ** 3  # heavy tails
    bound = torch.stack([row.abs().sort().values for row in samples]).quantile(0.95, dim=1).mean()
    clipped = clip_outliers(samples)
    assert clipped.abs().max() == pytest.approx(bound.item())
    inside = samples.abs() <= bound
    assert torch.equal(clipped[inside], samples[inside])
    assert torch.equal(clipped[~inside], bound * samples[~inside].sign())


def test_clipped_cka_is_the_cka_of_clipped_unit_rows_without_centering() -> None:
    samples_x, samples_y = paired_samples()
    samples_x, samples_y = samples_x**3 + 3.0, samples_y**3 - 1.0  # heavy tails, far from centered
    unit_x, unit_y = normalize(clip_outliers(samples_x), dim=-1), normalize(clip_outliers(samples_y), dim=-1)
    assert clipped_cka(samples_x, samples_y) == pytest.approx(cka(unit_x, unit_y), rel=1e-12)
    unclipped = cka(normalize(samples_x, dim=-1), normalize(samples_y, dim=-1))
    assert abs(clipped_cka(samples_x, samples_y) - unclipped) > 0.01


def test_unbiased_cka_does_not_depend_on_centering() -> None:
    """Shifting the features does not change the unbiased CKA.

    The unbiased HSIC ignores adding f(x) + f(y) + c to a kernel, and centering the features of a linear kernel adds
    such terms. Leaving out the centering therefore only matters through the row normalization.
    """
    samples_x, samples_y = paired_samples()
    shifted_x, shifted_y = samples_x + torch.randn(12, dtype=torch.float64), samples_y - 2.0
    assert cka(shifted_x, shifted_y) == pytest.approx(cka(samples_x, samples_y), rel=1e-10)


def test_cka_of_kernels_equals_cka_of_embeddings() -> None:
    samples_x, samples_y = paired_samples()
    assert cka_of_kernels(samples_x @ samples_x.T, samples_y @ samples_y.T) == pytest.approx(
        cka(samples_x, samples_y), rel=1e-10
    )


def test_kernel_neighbors_equal_nearest_neighbors_without_ties() -> None:
    samples_x, samples_y = paired_samples()
    generator = torch.Generator().manual_seed(0)
    neighbors = kernel_neighbors(samples_x @ samples_x.T, 5, generator)
    assert torch.equal(neighbors, nearest_neighbors(samples_x, 5))
    overlap = neighbor_overlap(neighbors, kernel_neighbors(samples_y @ samples_y.T, 5, generator))
    assert overlap == mutual_knn(samples_x, samples_y, 5)


def test_kernel_neighbors_break_ties_at_random() -> None:
    """Neighbors among tied kernel entries are drawn at random, not in index order. Clipped kernels have many ties."""
    kernel = torch.zeros(100, 100, dtype=torch.float64)
    kernel[:, 0] = 1.0  # sample 0 is the nearest neighbor of all samples, and all other similarities tie
    neighbors = kernel_neighbors(kernel, 3, torch.Generator().manual_seed(0))
    assert (neighbors[1:, 0] == 0).all()
    assert (neighbors != torch.arange(100)[:, None]).all()  # never the sample itself
    ties = neighbors[1:, 1:]
    assert ties.float().mean() > 20  # the index order would pick 1 and 2 (or 2 and 3) every time
    assert len(set(map(tuple, ties.tolist()))) > 90


def test_ordinal_indices_of_kernels_equal_those_of_embeddings() -> None:
    """With the negative distance as kernel, the same draws give the same TSI and QSI as the embedding scores.

    The QSI differs in one convention. A quadruplet with {i, j} = {k, l} is an exact tie. The kernel version counts it
    as one half, and the embedding version (two false strict comparisons) counts it as an agreement.
    """
    samples_x, samples_y = paired_samples()
    exact = "donot_use_mm_for_euclid_dist"  # the same distances as the embedding scores, bit for bit
    closeness_x = -torch.cdist(samples_x, samples_x, compute_mode=exact)
    closeness_y = -torch.cdist(samples_y, samples_y, compute_mode=exact)
    tsi = triplet_similarity_index(samples_x, samples_y, torch.Generator().manual_seed(1), 20_000)
    tsi_of_kernels = triplet_similarity_index_of_kernels(
        closeness_x, closeness_y, torch.Generator().manual_seed(1), 20_000
    )
    assert tsi_of_kernels == pytest.approx(tsi, rel=1e-7)  # the embedding variant averages in single precision

    generator = torch.Generator().manual_seed(1)
    i, j, k, l = (torch.randint(0, 200, (20_000,), generator=generator) for _ in range(4))  # noqa: E741
    valid = (i != j) & (k != l)
    same_pair = (((i == k) & (j == l)) | ((i == l) & (j == k)))[valid]
    assert same_pair.sum() > 0
    qsi = quadruplet_similarity_index(samples_x, samples_y, torch.Generator().manual_seed(1), 20_000)
    qsi_of_kernels = quadruplet_similarity_index_of_kernels(
        closeness_x, closeness_y, torch.Generator().manual_seed(1), 20_000
    )
    assert qsi_of_kernels == pytest.approx(qsi - 0.5 * same_pair.float().mean().item(), rel=1e-6)


def test_ties_count_half() -> None:
    first_x, second_x = torch.tensor([1.0, 1.0, 2.0, 0.0]), torch.tensor([0.0, 1.0, 1.0, 1.0])
    first_y, second_y = torch.tensor([1.0, 0.0, 1.0, 2.0]), torch.tensor([0.0, 1.0, 1.0, 1.0])
    # agreement, tie in x, tie in y, disagreement
    assert ordering_agreement(first_x, second_x, first_y, second_y) == (1 + 0.5 + 0.5 + 0) / 4
    constant = torch.ones(30, 30, dtype=torch.float64)
    samples = torch.randn(30, 4, dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    assert triplet_similarity_index_of_kernels(constant, samples @ samples.T, generator, 10_000) == 0.5
    assert quadruplet_similarity_index_of_kernels(constant, samples @ samples.T, generator, 10_000) == 0.5
