"""Tests of the budgeted MPOpt runs of Figure 7 against brute force and of the class-matching benchmark algebra."""

from __future__ import annotations

import itertools

import pytest
import torch

from modalities.qap_benchmark import cka_kernel, standardize_kernel
from unpaired_rosetta.ablation.qap_solvers import MPOptBudget, qap_cost, solve_with_bound
from unpaired_rosetta.qap import MPOptQAPSolver


def distance_kernels(
    size: int,
    seed: int = 0,
    noise: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardized distance kernels of two noisy views of the same points, the second in shuffled order."""
    generator = torch.Generator().manual_seed(seed)
    points = torch.randn(size, 5, dtype=torch.float64, generator=generator)
    other = points + noise * torch.randn(size, 5, dtype=torch.float64, generator=generator)
    other = other[torch.randperm(size, generator=generator)]
    return standardize_kernel(torch.cdist(points, points)), standardize_kernel(torch.cdist(other, other))


def gromov_wasserstein_problem(
    size: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vision_kernel, language_kernel = distance_kernels(size, seed)
    constant = (vision_kernel**2).sum() + (language_kernel**2).sum()
    return -vision_kernel, 2 * language_kernel, constant


def brute_force(
    cost1: torch.Tensor,
    cost2: torch.Tensor,
    constant: torch.Tensor,
) -> float:
    n = cost1.shape[0]
    return min(qap_cost(cost1, cost2, torch.tensor(p), constant.item()) for p in itertools.permutations(range(n)))


def is_permutation(
    permutation: torch.Tensor,
    size: int,
) -> bool:
    return torch.equal(permutation.sort().values, torch.arange(size))


def test_the_split_squared_loss_is_the_gromov_wasserstein_cost() -> None:
    vision_kernel, language_kernel = distance_kernels(9)
    cost1, cost2, constant = -vision_kernel, 2 * language_kernel, (vision_kernel**2).sum() + (language_kernel**2).sum()
    for seed in range(5):
        permutation = torch.randperm(9, generator=torch.Generator().manual_seed(seed))
        direct = ((vision_kernel - language_kernel[permutation][:, permutation]) ** 2).sum().item()
        assert qap_cost(cost1, cost2, permutation, constant.item()) == pytest.approx(direct, rel=1e-12)
        matrix = torch.eye(9, dtype=torch.float64)[permutation]  # P[i, pi(i)] = 1
        trace = torch.trace(cost1 @ matrix @ cost2.T @ matrix.T) + constant
        assert qap_cost(cost1, cost2, permutation, constant.item()) == pytest.approx(trace.item(), rel=1e-12)


def test_standardized_kernels() -> None:
    kernel = standardize_kernel(torch.rand(12, 12, dtype=torch.float64))
    off_diagonal = ~torch.eye(12, dtype=torch.bool)
    assert kernel.diagonal().abs().max() == 0
    assert kernel[off_diagonal].mean().abs() < 1e-12
    assert kernel[off_diagonal].std().item() == pytest.approx(1.0)
    centered = cka_kernel(torch.randn(7, 4, dtype=torch.float64))
    assert centered.mean(dim=1).abs().max() < 1e-12
    assert (centered * centered.T).sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("budget", [MPOptBudget(), MPOptBudget(20, 20, 50, 0.6, 50)])
def test_bound_and_cost_against_brute_force(
    budget: MPOptBudget,
) -> None:
    cost1, cost2, constant = gromov_wasserstein_problem(7)
    result = solve_with_bound(cost1, cost2, constant.item(), budget, seed=3)
    optimum = brute_force(cost1, cost2, constant)
    assert is_permutation(result.permutation, 7)
    assert result.cost == pytest.approx(qap_cost(cost1, cost2, result.permutation, constant.item()))
    assert result.bound <= optimum + 1e-6
    assert result.cost >= optimum - 1e-9
    assert 1 <= result.batches <= budget.max_batches
    assert result.seconds >= 0


def test_the_budget_reaches_the_solver_and_the_log_does_not_change_the_result() -> None:
    cost1, cost2, constant = gromov_wasserstein_problem(8, seed=1)
    budget = MPOptBudget(batch_size=5, greedy_generations=3, max_batches=7, stopping_p=0.6, stopping_k=100)
    result = solve_with_bound(cost1, cost2, constant.item(), budget, seed=11)
    assert result.batches == 7  # stopping_k = 100 never stops early
    solver = MPOptQAPSolver()
    solver.batch_size, solver.greedy_generations, solver.max_batches = 5, 3, 7
    solver.stopping_p, solver.stopping_k = 0.6, 100
    assert torch.equal(result.permutation, solver.solve(cost1, cost2, None, seed=11))
