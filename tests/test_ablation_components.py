"""Tests of the initializations, read-outs and refinements that the component ablation compares with ours."""

import itertools

import numpy as np
import pytest
import torch

from unpaired_rosetta.ablation.aligner import AblatedWassersteinProcrustes
from unpaired_rosetta.ablation.initializations import (
    ClusterMatching,
    GromovWasserstein,
    PCAHeuristic,
    SortedHeuristic,
    average_cross_covariance,
    best_two_opt,
    principal_directions,
)
from unpaired_rosetta.ablation.readouts import relative_representation_readout
from unpaired_rosetta.ablation.refinements import mini_vec2vec_refinement, moving_average
from unpaired_rosetta.ablation.sinkhorn import sinkhorn_projection
from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.geometric_initialization import GeometricInitialization
from unpaired_rosetta.linalg import center_and_normalize, polar
from unpaired_rosetta.randomness import Randomness
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes


def rotated_pair(
    num_samples: int = 300,
    dim: int = 6,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Anisotropic, skewed samples X and their rotated copies Y = X Q in shuffled order."""
    generator = torch.Generator().manual_seed(seed)
    samples_x = torch.randn(num_samples, dim, generator=generator).exp() * torch.linspace(3, 0.5, dim)
    samples_x = samples_x - samples_x.mean(dim=0)
    rotation = polar(torch.randn(dim, dim, generator=generator))
    order = torch.randperm(num_samples, generator=generator)
    return samples_x, (samples_x @ rotation)[order], rotation, order


def clustered_pair(
    num_samples: int = 400,
    dim: int = 8,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Well separated Gaussian clusters of uneven size and a rotated, shuffled copy."""
    generator = torch.Generator().manual_seed(seed)
    centers = 4 * torch.randn(6, dim, generator=generator)
    labels = torch.multinomial(
        torch.tensor([0.3, 0.25, 0.15, 0.12, 0.1, 0.08]), num_samples, replacement=True, generator=generator
    )
    samples_x = centers[labels] + 0.3 * torch.randn(num_samples, dim, generator=generator)
    rotation = polar(torch.randn(dim, dim, generator=generator))
    samples_y = (samples_x @ rotation)[torch.randperm(num_samples, generator=generator)]
    return samples_x, samples_y, rotation


def preprocessed(
    samples: torch.Tensor,
) -> torch.Tensor:
    return center_and_normalize(samples, samples.mean(dim=0, keepdim=True))


def test_principal_directions_are_orthonormal_and_positively_skewed() -> None:
    samples_x, _, _, _ = rotated_pair()
    directions = principal_directions(samples_x, 4)
    torch.testing.assert_close(directions @ directions.T, torch.eye(4), atol=1e-5, rtol=0)
    assert ((samples_x @ directions.T) ** 3).mean(dim=0).min() > 0


def test_pca_heuristic_recovers_a_rotation_of_skewed_data() -> None:
    samples_x, samples_y, rotation, _ = rotated_pair()
    correspondence = PCAHeuristic().restart(samples_x, samples_y)
    torch.testing.assert_close(polar(correspondence), rotation, atol=1e-3, rtol=0)


def test_sorted_heuristic_matches_rotated_copies() -> None:
    samples_x, samples_y, rotation, order = rotated_pair()
    correspondence = SortedHeuristic.restart(samples_x, samples_y)
    expected = samples_x.T @ samples_x / samples_x.shape[0] @ rotation  # every sample matched to its own copy
    torch.testing.assert_close(correspondence, expected, atol=1e-4, rtol=1e-4)


def test_random_initial_plan_is_a_coupling() -> None:
    generator = torch.Generator().manual_seed(0)
    log_kernel = torch.rand(7, 5, dtype=torch.float64, generator=generator).log()
    rows, columns = torch.full((7,), 1 / 7, dtype=torch.float64), torch.full((5,), 1 / 5, dtype=torch.float64)
    plan = sinkhorn_projection(log_kernel, rows, columns, num_iterations=1000)
    torch.testing.assert_close(plan.sum(dim=1), rows)
    torch.testing.assert_close(plan.sum(dim=0), columns)


@pytest.mark.parametrize("initial_plan", ["uniform", "random"])
def test_gromov_wasserstein_initialization_runs(
    initial_plan: str,
) -> None:
    samples_x, samples_y, _ = clustered_pair(num_samples=60)
    correspondence = GromovWasserstein(initial_plan).restart(
        preprocessed(samples_x), preprocessed(samples_y), Randomness(0)
    )
    assert correspondence.shape == (8, 8) and torch.isfinite(correspondence).all()


def test_best_two_opt_returns_a_local_optimum() -> None:
    generator = np.random.RandomState(1)
    cost_x, cost_y = torch.as_tensor(generator.randn(7, 7)), torch.as_tensor(generator.randn(7, 7))
    permutation = best_two_opt(cost_x, cost_y, seeds=[1, 2, 3]).numpy()

    def objective(
        p: np.ndarray,
    ) -> float:
        return float(np.sum(cost_x.numpy() * cost_y.numpy()[p][:, p]))

    for i, j in itertools.combinations(range(7), 2):
        swapped = permutation.copy()
        swapped[i], swapped[j] = swapped[j], swapped[i]
        assert objective(swapped) >= objective(permutation) - 1e-12


def test_cluster_matching_with_mpopt_equals_the_geometric_initialization() -> None:
    samples_x, samples_y, _ = clustered_pair()
    samples_x, samples_y = preprocessed(samples_x), preprocessed(samples_y)
    ours = GeometricInitialization(num_clusters=5, num_restarts=3, batch_size=300)(
        samples_x, samples_y, None, None, Randomness(3)
    )
    matched = ClusterMatching(num_clusters=5, num_restarts=3, batch_size=300)(samples_x, samples_y, Randomness(3))
    assert torch.equal(average_cross_covariance(matched), ours)


def test_cluster_matching_recovers_the_rotation_from_matched_centers() -> None:
    samples_x, samples_y, rotation = clustered_pair()
    for solver in ("MPOpt", "2-opt"):
        matched = ClusterMatching(num_clusters=6, solver=solver, num_restarts=2)(samples_x, samples_y, Randomness(0))
        for centers_x, centers_y in matched:
            torch.testing.assert_close(centers_x @ rotation, centers_y, atol=0.2, rtol=0)


def test_restarts_are_a_prefix_of_longer_runs() -> None:
    samples_x, samples_y, _ = clustered_pair()
    short = ClusterMatching(num_clusters=4, num_restarts=2, batch_size=300)(samples_x, samples_y, Randomness(5))
    long = ClusterMatching(num_clusters=4, num_restarts=4, batch_size=300)(samples_x, samples_y, Randomness(5))
    assert torch.equal(average_cross_covariance(long[:2]), average_cross_covariance(short))


def test_relative_representation_readout_recovers_a_rotation() -> None:
    samples_x, samples_y, rotation = clustered_pair(dim=4)  # the six clusters span all four dimensions
    matched = ClusterMatching(num_clusters=6, num_restarts=2)(samples_x, samples_y, Randomness(0))
    anchors_x, anchors_y = torch.cat([a for a, _ in matched]), torch.cat([b for _, b in matched])
    weight = relative_representation_readout(samples_x, samples_y, anchors_x, anchors_y)
    torch.testing.assert_close(weight, rotation, atol=0.1, rtol=0)


def test_moving_average_stays_semi_orthogonal() -> None:
    generator = torch.Generator().manual_seed(0)
    weight = polar(torch.randn(6, 4, generator=generator))
    updated = moving_average(weight, torch.randn(6, 4, generator=generator))
    torch.testing.assert_close(updated.T @ updated, torch.eye(4), atol=1e-5, rtol=0)


def test_mini_vec2vec_refinement_keeps_a_correct_map() -> None:
    samples_x, samples_y, rotation = clustered_pair(num_samples=1200, dim=5)
    samples_x, samples_y = preprocessed(samples_x), preprocessed(samples_y)
    refined = mini_vec2vec_refinement(samples_x, samples_y, rotation, Randomness(0))
    torch.testing.assert_close(refined, rotation, atol=0.05, rtol=0)


def test_unablated_aligner_is_our_method() -> None:
    samples_x, samples_y, _ = clustered_pair()
    hyperparameters = dict(num_clusters=5, num_restarts=3, batch_size=250, num_iterations=4)
    ours = WassersteinProcrustes(**hyperparameters).fit(samples_x, samples_y, randomness=Randomness(7))
    ablated = AblatedWassersteinProcrustes(**hyperparameters).fit(samples_x, samples_y, randomness=Randomness(7))
    assert torch.equal(ours.weight, ablated.weight)


def test_matched_centers_do_not_change_our_pipeline() -> None:
    """The relative-representation run clusters through ``ClusterMatching`` and gets the same ``M`` as ours."""
    samples_x, samples_y, _ = clustered_pair()
    hyperparameters = dict(num_clusters=5, num_restarts=3, batch_size=250, num_iterations=0)
    ours = WassersteinProcrustes(**hyperparameters).fit(samples_x, samples_y, randomness=Randomness(7))
    ablated = AblatedWassersteinProcrustes(readout="relative representations", refinement="none", **hyperparameters)
    with torch_threads():
        samples = ablated.preprocess(samples_x, samples_y)
        randomness = Randomness(7)
        correspondence, _ = ablated.initialize(*samples, randomness)
        weight = ablated.readout(*samples, correspondence, None, None, randomness)
    assert torch.equal(weight, ours.weight)


@pytest.mark.parametrize(
    "variant",
    [
        dict(initialization="PCA", refinement="none"),
        dict(initialization="sorted", refinement="none"),
        dict(initialization="GW (uniform)", refinement="none"),
        dict(initialization="2-opt", refinement="none"),
        dict(readout="direct", refinement="none"),
        dict(readout="relative representations", refinement="none"),
        dict(refinement="mini-vec2vec"),
    ],
)
def test_every_variant_fits(
    variant: dict[str, str],
) -> None:
    samples_x, samples_y, _ = clustered_pair(num_samples=600)
    aligner = AblatedWassersteinProcrustes(**variant, num_clusters=5, num_restarts=2, batch_size=300, num_iterations=2)
    aligner.fit(samples_x, samples_y, randomness=Randomness(0))
    assert aligner.similarity(samples_x[:10], samples_y[:20]).shape == (10, 20)
