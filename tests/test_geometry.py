"""Tests of the geometry scores against their dense textbook definitions."""

import itertools

import pytest
import torch

from unpaired_rosetta.geometry import cka, mutual_knn, quadruplet_similarity_index, triplet_similarity_index


def paired_samples(
    num_samples: int = 300,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    samples_x = torch.randn(num_samples, 12, dtype=torch.float64, generator=generator)
    samples_y = samples_x @ torch.randn(12, 9, dtype=torch.float64, generator=generator) + torch.randn(
        num_samples, 9, dtype=torch.float64, generator=generator
    )
    return samples_x, samples_y


def unbiased_hsic(
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
) -> torch.Tensor:
    n = kernel_x.shape[0]
    kernel_x, kernel_y = kernel_x.clone().fill_diagonal_(0), kernel_y.clone().fill_diagonal_(0)
    ones = torch.ones(n, dtype=torch.float64)
    return (
        torch.trace(kernel_x @ kernel_y)
        + (ones @ kernel_x @ ones) * (ones @ kernel_y @ ones) / ((n - 1) * (n - 2))
        - 2 * (ones @ kernel_x @ kernel_y @ ones) / (n - 2)
    ) / (n * (n - 3))


def test_cka_is_the_unbiased_estimator() -> None:
    samples_x, samples_y = paired_samples()
    kernel_x, kernel_y = samples_x @ samples_x.T, samples_y @ samples_y.T
    expected = (
        unbiased_hsic(kernel_x, kernel_y)
        / (unbiased_hsic(kernel_x, kernel_x) * unbiased_hsic(kernel_y, kernel_y)).sqrt()
    )
    assert cka(samples_x, samples_y, block_size=64) == pytest.approx(expected.item(), rel=1e-10)
    assert cka(samples_x, samples_x) == pytest.approx(1.0)


def test_cka_is_invariant_to_rotations_and_scaling() -> None:
    samples_x, samples_y = paired_samples()
    rotation = torch.linalg.qr(torch.randn(9, 9, dtype=torch.float64))[0]
    assert cka(samples_x, 3 * samples_y @ rotation) == pytest.approx(cka(samples_x, samples_y), rel=1e-10)


def test_mutual_knn_by_definition() -> None:
    samples_x, samples_y = paired_samples(200)

    def neighbor_sets(
        samples: torch.Tensor,
    ) -> list[set[int]]:
        similarities = (samples @ samples.T).fill_diagonal_(-float("inf"))
        return [set(row.tolist()) for row in similarities.topk(10, dim=1).indices]

    expected = sum(len(a & b) / 10 for a, b in zip(neighbor_sets(samples_x), neighbor_sets(samples_y))) / 200
    assert mutual_knn(samples_x, samples_y) == pytest.approx(expected)
    assert mutual_knn(samples_x, samples_x) == 1.0


def test_ordinal_indices_by_exhaustive_enumeration() -> None:
    samples_x, samples_y = paired_samples(12)
    distance_x, distance_y = torch.cdist(samples_x, samples_x), torch.cdist(samples_y, samples_y)
    triplets = [(i, j, k) for i, j, k in itertools.permutations(range(12), 3)]
    exact_tsi = sum(
        (distance_x[i, j] < distance_x[i, k]) == (distance_y[i, j] < distance_y[i, k]) for i, j, k in triplets
    ) / len(triplets)
    quadruplets = [(i, j, k, l) for i, j, k, l in itertools.product(range(12), repeat=4) if i != j and k != l]
    exact_qsi = sum(
        (distance_x[i, j] < distance_x[k, l]) == (distance_y[i, j] < distance_y[k, l]) for i, j, k, l in quadruplets
    ) / len(quadruplets)
    generator = torch.Generator().manual_seed(0)
    assert triplet_similarity_index(samples_x, samples_y, generator, 200_000) == pytest.approx(
        exact_tsi.item(), abs=0.01
    )
    assert quadruplet_similarity_index(samples_x, samples_y, generator, 200_000) == pytest.approx(
        exact_qsi.item(), abs=0.01
    )
