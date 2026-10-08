"""Tests of the baselines that have no official code. The linear and orthogonal maps fitted on the known pairs and the
identity map are checked against their definitions, NumPy and SciPy."""

import numpy as np
import pytest
import torch
from scipy.linalg import orthogonal_procrustes

from unpaired_rosetta.baselines.identity import Identity
from unpaired_rosetta.baselines.linear import LinearMap
from unpaired_rosetta.baselines.orthogonal import OrthogonalMap
from unpaired_rosetta.linalg import center_and_normalize, polar
from unpaired_rosetta.randomness import Randomness


def problem(
    num_pairs: int,
    dim_x: int = 12,
    dim_y: int = 16,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpaired training sets, known pairs related by a known semi-orthogonal map, and validation samples."""
    generator = torch.Generator().manual_seed(seed)
    rotation = polar(torch.randn(dim_x, dim_y, dtype=torch.float64, generator=generator))
    draw = lambda num: torch.randn(num, dim_x, dtype=torch.float64, generator=generator) + 0.5  # noqa: E731
    samples_x, samples_y = draw(300), draw(300) @ rotation
    paired_x = draw(num_pairs)
    paired_y = paired_x @ rotation + 0.01 * torch.randn(num_pairs, dim_y, dtype=torch.float64, generator=generator)
    val_x = draw(50)
    return samples_x, samples_y, paired_x, paired_y, val_x, val_x @ rotation


@pytest.mark.parametrize("num_pairs", [3, 12, 40])
def test_linear_map_is_the_minimum_norm_least_squares_solution(
    num_pairs: int,
) -> None:
    samples_x, samples_y, paired_x, paired_y, _, _ = problem(num_pairs)
    aligner = LinearMap().fit(samples_x, samples_y, paired_x, paired_y, Randomness(0))
    inputs = center_and_normalize(paired_x, samples_x.mean(0, keepdim=True)).numpy()
    targets = center_and_normalize(paired_y, samples_y.mean(0, keepdim=True)).numpy()
    expected = np.linalg.lstsq(inputs, targets, rcond=None)[
        0
    ]  # gelsd gives the minimum norm solution also for fewer pairs than dimensions
    np.testing.assert_allclose(aligner.weight.numpy(), expected, atol=1e-10)


def test_orthogonal_map_equals_scipy_procrustes() -> None:
    samples_x, samples_y, paired_x, paired_y, _, _ = problem(40, dim_x=12, dim_y=12)
    aligner = OrthogonalMap().fit(samples_x, samples_y, paired_x, paired_y, Randomness(0))
    inputs = center_and_normalize(paired_x, samples_x.mean(0, keepdim=True)).numpy()
    targets = center_and_normalize(paired_y, samples_y.mean(0, keepdim=True)).numpy()
    np.testing.assert_allclose(aligner.weight.numpy(), orthogonal_procrustes(inputs, targets)[0], atol=1e-10)


@pytest.mark.parametrize("num_pairs", [3, 40])
def test_orthogonal_map_is_semi_orthogonal_and_maximizes_the_trace(
    num_pairs: int,
) -> None:
    samples_x, samples_y, paired_x, paired_y, _, _ = problem(num_pairs)
    aligner = OrthogonalMap().fit(samples_x, samples_y, paired_x, paired_y, Randomness(0))
    weight = aligner.weight
    torch.testing.assert_close(weight @ weight.T, torch.eye(12, dtype=torch.float64))
    inputs = center_and_normalize(paired_x, samples_x.mean(0, keepdim=True))
    targets = center_and_normalize(paired_y, samples_y.mean(0, keepdim=True))
    best = torch.trace(inputs @ weight @ targets.T)
    generator = torch.Generator().manual_seed(1)
    for _ in range(20):
        other = polar(weight + 0.1 * torch.randn(weight.shape, dtype=torch.float64, generator=generator))
        assert torch.trace(inputs @ other @ targets.T) <= best + 1e-12


@pytest.mark.parametrize("aligner_class", [LinearMap, OrthogonalMap])
def test_maps_score_the_preprocessed_mapped_queries_and_recover_the_planted_map(
    aligner_class: type[LinearMap | OrthogonalMap],
) -> None:
    samples_x, samples_y, paired_x, paired_y, val_x, val_y = problem(40)
    aligner = aligner_class().fit(samples_x, samples_y, paired_x, paired_y, Randomness(0))
    queries = center_and_normalize(val_x, samples_x.mean(0, keepdim=True))
    keys = center_and_normalize(val_y, samples_y.mean(0, keepdim=True))
    scores = aligner.similarity(val_x, val_y)
    torch.testing.assert_close(scores, queries @ aligner.weight @ keys.T)
    assert (scores.argmax(dim=1) == torch.arange(50)).all()


@pytest.mark.parametrize("aligner_class", [LinearMap, OrthogonalMap])
def test_maps_without_pairs_score_at_chance(
    aligner_class: type[LinearMap | OrthogonalMap],
) -> None:
    samples_x, samples_y, _, _, val_x, val_y = problem(0)
    aligner = aligner_class().fit(samples_x, samples_y, None, None, Randomness(0))
    assert (aligner.similarity(val_x, val_y) == 0).all()


def test_identity_is_the_inner_product_of_the_inputs() -> None:
    generator = torch.Generator().manual_seed(0)
    samples_x, samples_y, val_x, val_y = (torch.randn(20, 6, generator=generator) + 1 for _ in range(4))
    aligner = Identity().fit(samples_x, samples_y, None, None, Randomness(0))
    assert torch.equal(aligner.similarity(val_x, val_y), val_x @ val_y.T)
    centered = Identity(centered=True).fit(samples_x, samples_y, None, None, Randomness(0))
    queries = center_and_normalize(val_x, samples_x.mean(0, keepdim=True))
    keys = center_and_normalize(val_y, samples_y.mean(0, keepdim=True))
    expected = queries @ keys.T
    assert torch.equal(centered.similarity(val_x, val_y), expected)


def test_identity_needs_equal_widths() -> None:
    with pytest.raises(ValueError):
        Identity().fit(torch.zeros(4, 3), torch.zeros(4, 5), None, None, Randomness(0))
