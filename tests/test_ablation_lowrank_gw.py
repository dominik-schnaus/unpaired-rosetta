"""Tests of the low-rank Gromov-Wasserstein ablation (Table 2). They check that the plans are feasible, the loss
against its definition and POT, the mirror descent, and cluster matching on problems with a known answer."""

import itertools

import numpy as np
import ot
import pytest
import torch

from experiments.lowrank_gw import costs
from modalities import lowrank_benchmarks
from unpaired_rosetta.ablation import lowrank_gw
from unpaired_rosetta.randomness import Randomness


def planted_blobs(
    num_per_cluster: int = 25,
    num_clusters: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Well separated blobs of different sizes and their shuffled image under a random isometry."""
    generator = torch.Generator().manual_seed(seed)
    centers = 10 * torch.randn(num_clusters, 3, generator=generator, dtype=torch.float64)
    sizes = [num_per_cluster * (k + 1) for k in range(num_clusters)]
    samples_x = torch.cat(
        [
            center + 0.1 * torch.randn(size, 3, generator=generator, dtype=torch.float64)
            for center, size in zip(centers, sizes)
        ]
    )
    labels = torch.cat([torch.full((size,), k) for k, size in enumerate(sizes)])
    samples_y, permutation = lowrank_benchmarks.shuffle(
        lowrank_benchmarks.isometric_embedding(samples_x, 5, generator), generator
    )
    return samples_x - samples_x.mean(0), samples_y - samples_y.mean(0), permutation, labels


def assert_feasible(
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_x: int,
    num_y: int,
    tolerance: float = 1e-10,
) -> None:
    q, r, g = factors
    assert (q >= 0).all() and (r >= 0).all() and (g > 0).all()
    torch.testing.assert_close(q.sum(dim=1), lowrank_gw.uniform(num_x), atol=tolerance, rtol=0)
    torch.testing.assert_close(r.sum(dim=1), lowrank_gw.uniform(num_y), atol=tolerance, rtol=0)
    torch.testing.assert_close(q.sum(dim=0), g, atol=tolerance, rtol=0)
    torch.testing.assert_close(r.sum(dim=0), g, atol=tolerance, rtol=0)
    plan = lowrank_gw.dense_plan(factors)
    torch.testing.assert_close(plan.sum(dim=1), lowrank_gw.uniform(num_x), atol=tolerance, rtol=0)
    torch.testing.assert_close(plan.sum(dim=0), lowrank_gw.uniform(num_y), atol=tolerance, rtol=0)


def test_gw_loss_equals_its_definition() -> None:
    generator = torch.Generator().manual_seed(0)
    cost_x, cost_y = (
        torch.rand(5, 5, dtype=torch.float64, generator=generator),
        torch.rand(4, 4, dtype=torch.float64, generator=generator),
    )
    cost_x, cost_y = cost_x + cost_x.T, cost_y + cost_y.T
    plan = torch.rand(5, 4, dtype=torch.float64, generator=generator)
    plan = plan / plan.sum()
    expected = sum(
        (cost_x[i, k] - cost_y[j, l]) ** 2 * plan[i, j] * plan[k, l]
        for i, k in itertools.product(range(5), repeat=2)
        for j, l in itertools.product(range(4), repeat=2)
    )
    assert lowrank_gw.gw_loss(cost_x, cost_y, plan) == pytest.approx(expected.item(), rel=1e-12)


def test_gw_loss_equals_pot() -> None:
    """The loss equals the square-loss Gromov-Wasserstein objective of POT for the same plan."""
    samples_x, samples_y, _, _ = planted_blobs()
    cost_x, cost_y = lowrank_gw.squared_euclidean_cost(samples_x), lowrank_gw.squared_euclidean_cost(samples_y)
    plan = torch.outer(lowrank_gw.uniform(len(samples_x)), lowrank_gw.uniform(len(samples_y)))
    constant, h_x, h_y = ot.gromov.init_matrix(cost_x.numpy(), cost_y.numpy(), plan.sum(1).numpy(), plan.sum(0).numpy())
    expected = ot.gromov.gwloss(constant, h_x, h_y, plan.numpy())
    assert lowrank_gw.gw_loss(cost_x, cost_y, plan) == pytest.approx(float(expected), rel=1e-12)


def test_plan_foscttm_of_the_true_matching_is_zero() -> None:
    samples_x, samples_y, permutation, _ = planted_blobs()
    plan = torch.zeros(len(samples_x), len(samples_y), dtype=torch.float64)
    plan[torch.arange(len(samples_x)), permutation] = 1.0 / len(samples_x)
    assert lowrank_gw.plan_foscttm(plan, samples_y, permutation) == 0.0
    independent = torch.full_like(plan, 1.0 / plan.numel())
    assert lowrank_gw.plan_foscttm(independent, samples_y, permutation) > 0.3


def test_geodesic_cost_is_a_normalized_metric() -> None:
    samples = torch.randn(80, 6, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    cost = lowrank_gw.geodesic_cost(samples, num_neighbors=10)
    assert cost.max() == 1.0
    torch.testing.assert_close(cost, cost.T)
    assert (cost.diagonal() == 0).all()


def test_benchmarks_have_the_documented_correspondence() -> None:
    """``Y[permutation[i]]`` is the isometric image of ``X[i]``, so all pairwise distances agree."""
    for name in ["blobs_k10", "curve", "uniform"]:
        samples_x, samples_y, permutation = lowrank_benchmarks.load(name, seed=1, num_samples=200)
        torch.testing.assert_close(
            torch.cdist(samples_x, samples_x),
            torch.cdist(samples_y[permutation], samples_y[permutation]),
            atol=1e-5,
            rtol=1e-7,
        )


@pytest.mark.parametrize("initialization", lowrank_gw.INITIALIZATIONS)
def test_initial_plans(
    initialization: str,
) -> None:
    """Every starting plan has the row marginals ``a`` and ``b``. All but the random one are feasible low-rank plans.

    The random start of POT is not feasible because its ``Q`` and ``R`` do not share the column sums ``g``, as in
    Scetbon et al. The trivial start is built in single precision. After the lower bound of POT,
    2000 Sinkhorn iterations meet the column sums only to about 1e-5, because the POT factors have entries near 1e-17.
    """
    samples_x, samples_y, _, _ = planted_blobs()
    q, r, g = lowrank_gw.initial_factors(samples_x, samples_y, 4, initialization, Randomness(0))
    tolerance = {"trivial": 1e-6, "lower bound": 1e-4}.get(initialization, 1e-10)
    torch.testing.assert_close(q.sum(dim=1), lowrank_gw.uniform(len(samples_x)), atol=tolerance, rtol=0)
    torch.testing.assert_close(r.sum(dim=1), lowrank_gw.uniform(len(samples_y)), atol=tolerance, rtol=0)
    if initialization != "random":
        assert_feasible((q, r, g), len(samples_x), len(samples_y), tolerance)


def test_factors_from_clusters_follow_the_clusters() -> None:
    labels_x = torch.tensor([0, 0, 1, 1, 1, 2])
    labels_y = torch.tensor([2, 0, 0, 1, 1, 1, 1])
    factors = lowrank_gw.factors_from_clusters(labels_x, labels_y, 3)
    assert_feasible(factors, 6, 7)
    q, r, g = factors
    torch.testing.assert_close(g, ((torch.tensor([2, 3, 1]) / 6 + torch.tensor([2, 4, 1]) / 7) / 2).double())
    assert (q.argmax(dim=1) == labels_x).all() and (r.argmax(dim=1) == labels_y).all()


def test_matched_clusters_recover_a_planted_matching() -> None:
    """With k-means + QAP, the plan moves almost all mass between matching clusters. Plain k-means does not."""
    samples_x, samples_y, permutation, labels = planted_blobs()
    labels_y = torch.empty_like(labels)
    labels_y[permutation] = labels
    same_cluster = (labels[:, None] == labels_y[None, :]).double()
    matched = lowrank_gw.dense_plan(lowrank_gw.initial_factors(samples_x, samples_y, 4, "k-means + QAP", Randomness(0)))
    assert (matched * same_cluster).sum() > 0.99
    unmatched = lowrank_gw.dense_plan(lowrank_gw.initial_factors(samples_x, samples_y, 4, "k-means", Randomness(0)))
    assert (unmatched * same_cluster).sum() < 0.99


@pytest.mark.parametrize("initialization", ["random", "k-means"])
def test_mirror_descent_keeps_feasibility_and_improves(
    initialization: str,
) -> None:
    samples_x, samples_y, _, _ = planted_blobs()
    cost_x, cost_y = costs("blobs_k10", samples_x, samples_y)
    factors = lowrank_gw.initial_factors(samples_x, samples_y, 4, initialization, Randomness(0))
    loss_before = lowrank_gw.gw_loss(cost_x, cost_y, lowrank_gw.dense_plan(factors))
    final, history = lowrank_gw.mirror_descent(factors, cost_x, cost_y, num_iterations=50)
    assert_feasible(final, len(samples_x), len(samples_y))
    assert history[-1] >= max(history[:-1]) - 1e-9  # the best iterate is returned
    assert lowrank_gw.gw_loss(cost_x, cost_y, lowrank_gw.dense_plan(final)) <= loss_before + 1e-9
    assert np.isfinite(history).all()
