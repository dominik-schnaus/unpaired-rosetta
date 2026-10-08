"""Tests of the method against the equations of the paper, brute force and SciPy."""

import ctypes
import itertools
import os

import numpy as np
import pytest
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment

from unpaired_rosetta.assignment import batched_hungarian_matching, hungarian_matching
from unpaired_rosetta.geometric_initialization import (
    GeometricInitialization,
    centered_kernel,
    cka_matching_costs,
)
from unpaired_rosetta.linalg import center_and_normalize, polar
from unpaired_rosetta.qap import MPOptQAPSolver, qap_objective
from unpaired_rosetta.randomness import SEEDS, Randomness, derive_seeds
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes, procrustes


def all_permutations(
    size: int,
) -> list[torch.Tensor]:
    return [torch.tensor(permutation) for permutation in itertools.permutations(range(size))]


def test_the_seeds_of_the_paper() -> None:
    assert derive_seeds(42, 5) == [734796314, 576165995, 2197670066, 839703249, 2584932063]
    assert SEEDS == derive_seeds(42, 5)


def test_randomness_reproduces_the_global_generators() -> None:
    torch.manual_seed(SEEDS[0])
    np.random.seed(SEEDS[0])
    expected = (torch.randperm(1000), torch.randint(0, 2**32 - 1, (1,)).item(), np.random.randint(0, 2**31 - 1))
    randomness = Randomness(SEEDS[0])
    assert torch.equal(randomness.permutation(1000), expected[0])
    assert randomness.solver_seed() == expected[1]
    assert randomness.kmeans_seed() == expected[2]


@pytest.mark.parametrize("shape", [(20, 20), (8, 20), (20, 8)])
def test_polar_solves_orthogonal_procrustes(
    shape: tuple[int, int],
) -> None:
    generator = torch.Generator().manual_seed(0)
    samples_x = torch.randn(100, shape[0], dtype=torch.float64, generator=generator)
    samples_y = torch.randn(100, shape[1], dtype=torch.float64, generator=generator)
    weight = polar(samples_x.T @ samples_y)
    if shape[0] == shape[1]:
        expected, _ = orthogonal_procrustes(samples_x.numpy(), samples_y.numpy())
        torch.testing.assert_close(weight, torch.as_tensor(expected))
    small = min(shape)
    gram = weight.T @ weight if shape[0] >= shape[1] else weight @ weight.T
    torch.testing.assert_close(gram, torch.eye(small, dtype=torch.float64))  # semi-orthogonal
    # No other semi-orthogonal matrix has a larger Tr(X W Y^T).
    objective = torch.trace(samples_x @ weight @ samples_y.T)
    for _ in range(20):
        random_weight = polar(torch.randn(*shape, dtype=torch.float64, generator=generator))
        assert torch.trace(samples_x @ random_weight @ samples_y.T) <= objective


def test_center_and_normalize() -> None:
    samples = torch.randn(50, 7) + 3
    mean = samples.mean(dim=0, keepdim=True)
    processed = center_and_normalize(samples, mean)
    torch.testing.assert_close(processed.norm(dim=1), torch.ones(50))
    torch.testing.assert_close(processed, (samples - mean) / (samples - mean).norm(dim=1, keepdim=True))


def test_procrustes_weights_pairs_like_the_pseudo_pairs() -> None:
    generator = torch.Generator().manual_seed(1)
    matched_x, matched_y = torch.randn(40, 5, generator=generator), torch.randn(40, 6, generator=generator)
    paired_x, paired_y = torch.randn(4, 5, generator=generator), torch.randn(4, 6, generator=generator)
    expected = polar(matched_x.T @ matched_y + (40 / 4) * paired_x.T @ paired_y)
    torch.testing.assert_close(procrustes(matched_x, matched_y, paired_x, paired_y), expected)
    torch.testing.assert_close(procrustes(matched_x, matched_y, None, None), polar(matched_x.T @ matched_y))


def test_hungarian_matching_maximizes_the_total_score() -> None:
    scores = torch.randn(30, 40, generator=torch.Generator().manual_seed(2))
    rows, columns = hungarian_matching(scores)
    expected_rows, expected_columns = linear_sum_assignment(scores.numpy(), maximize=True)
    assert torch.equal(rows, torch.as_tensor(expected_rows))
    assert torch.equal(columns, torch.as_tensor(expected_columns))


@pytest.mark.parametrize("num_x, num_y", [(1000, 1000), (1003, 998), (250, 700)])
def test_batched_hungarian_matching_solves_every_block(
    num_x: int,
    num_y: int,
) -> None:
    generator = torch.Generator().manual_seed(3)
    queries, keys = torch.randn(num_x, 16, generator=generator), torch.randn(num_y, 16, generator=generator)
    block_size = 300
    rows, columns = batched_hungarian_matching(queries, keys, block_size)
    assert len(rows) == min(num_x, num_y)
    assert len(set(rows.tolist())) == len(rows) and len(set(columns.tolist())) == len(columns)
    for start in range(0, min(num_x, num_y), block_size):
        in_block = (rows >= start) & (rows < start + block_size)
        assert ((columns[in_block] >= start) & (columns[in_block] < start + block_size)).all()
        block = (queries[start : start + block_size] @ keys[start : start + block_size].T).double().numpy()
        expected_rows, expected_columns = linear_sum_assignment(block, maximize=True)
        assert torch.equal(rows[in_block], torch.as_tensor(expected_rows) + start)
        assert torch.equal(columns[in_block], torch.as_tensor(expected_columns) + start)
    parallel = batched_hungarian_matching(queries, keys, block_size, num_threads=3)
    assert torch.equal(parallel[0], rows) and torch.equal(parallel[1], columns)


def brute_force_minimum(
    cost_x: torch.Tensor,
    cost_y: torch.Tensor,
    linear: torch.Tensor | None = None,
) -> float:
    objectives = [
        qap_objective(cost_x, cost_y, permutation, linear) for permutation in all_permutations(cost_x.shape[0])
    ]
    return min(objectives)


@pytest.mark.parametrize("seed", range(5))
def test_mpopt_finds_the_optimum_of_small_problems(
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    centers_x, centers_y = torch.randn(7, 5, generator=generator), torch.randn(7, 9, generator=generator)
    paired_x, paired_y = torch.randn(3, 5, generator=generator), torch.randn(3, 9, generator=generator)
    for pairs in [(None, None), (paired_x, paired_y)]:
        cost_x, cost_y, linear = cka_matching_costs(
            centers_x.double(), centers_y.double(), *[p if p is None else p.double() for p in pairs]
        )
        permutation = MPOptQAPSolver().solve(cost_x, cost_y, linear, seed=seed)
        assert qap_objective(cost_x, cost_y, permutation, linear) == pytest.approx(
            brute_force_minimum(cost_x, cost_y, linear)
        )


def test_centered_kernel_is_h_k_h() -> None:
    samples = torch.randn(9, 4, dtype=torch.float64)
    centering = torch.eye(9, dtype=torch.float64) - 1 / 9
    torch.testing.assert_close(centered_kernel(samples), centering @ samples @ samples.T @ centering)


def clustered_spaces(
    num_samples: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two spaces with a known relation. Y is a rotated, partly projected copy of X plus noise."""
    generator = torch.Generator().manual_seed(seed)
    centers = 3 * torch.randn(12, 16, generator=generator)
    samples = centers[torch.randint(0, 12, (num_samples,), generator=generator)] + torch.randn(
        num_samples, 16, generator=generator
    )
    rotation = polar(torch.randn(16, 24, generator=generator))
    return samples, samples @ rotation + 0.05 * torch.randn(num_samples, 24, generator=generator)


def test_parallel_restarts_give_identical_initializations() -> None:
    samples_x, samples_y = clustered_spaces(600)
    initialization = GeometricInitialization(num_clusters=8, num_restarts=4, batch_size=300)
    sequential = initialization(samples_x, samples_y, None, None, Randomness(0))
    initialization.num_workers = 2
    parallel = initialization(samples_x, samples_y, None, None, Randomness(0))
    assert torch.equal(sequential, parallel)


def test_same_seed_gives_the_same_map() -> None:
    samples_x, samples_y = clustered_spaces(800)
    first = WassersteinProcrustes(num_clusters=8, num_restarts=3, batch_size=200, num_iterations=5).fit(
        samples_x, samples_y, randomness=1
    )
    second = WassersteinProcrustes(num_clusters=8, num_restarts=3, batch_size=200, num_iterations=5, num_threads=4).fit(
        samples_x, samples_y, randomness=1
    )
    assert torch.equal(first.weight, second.weight)


def test_recovers_a_planted_map_from_disjoint_halves() -> None:
    from unpaired_rosetta.evaluation import disjoint_split, retrieval_metrics

    samples_x, samples_y = clustered_spaces(4000)
    randomness = Randomness(3)
    indices_x, indices_y = disjoint_split(3000, randomness)
    aligner = WassersteinProcrustes(num_clusters=12, num_restarts=5, batch_size=1000, num_iterations=10)
    aligner.fit(samples_x[indices_x], samples_y[indices_y], randomness=randomness)
    metrics = retrieval_metrics(aligner, samples_x[3000:], samples_y[3000:], Randomness(3))
    assert metrics["foscttm"] < 0.02
    weight = aligner.weight
    torch.testing.assert_close(weight @ weight.T, torch.eye(16), atol=1e-5, rtol=0)


def test_pairs_are_used_and_improve_a_hard_problem() -> None:
    from unpaired_rosetta.evaluation import disjoint_split, retrieval_metrics

    samples_x, samples_y = clustered_spaces(3000, seed=5)
    samples_y = samples_y + 1.5 * torch.randn(samples_y.shape, generator=torch.Generator().manual_seed(9))
    randomness = Randomness(4)
    indices_x, indices_y = disjoint_split(2000, randomness)
    pairs = torch.arange(2000, 2050)
    common = dict(num_clusters=12, num_restarts=3, batch_size=500, num_iterations=5)
    unpaired = WassersteinProcrustes(**common).fit(samples_x[indices_x], samples_y[indices_y], randomness=4)
    paired = WassersteinProcrustes(**common).fit(
        samples_x[indices_x], samples_y[indices_y], samples_x[pairs], samples_y[pairs], randomness=4
    )
    assert not torch.equal(unpaired.weight, paired.weight)
    validation = (samples_x[2500:], samples_y[2500:])
    assert (
        retrieval_metrics(paired, *validation, Randomness(0))["foscttm"]
        <= retrieval_metrics(unpaired, *validation, Randomness(0))["foscttm"] + 1e-3
    )


def test_global_generators_continue_the_run_streams() -> None:
    """Code that draws from the global generators sees the same stream as with the old global seeding."""
    import random

    torch.manual_seed(SEEDS[1])
    np.random.seed(SEEDS[1])
    random.seed(SEEDS[1])
    expected = (torch.randperm(50), torch.randperm(20), np.random.rand(3), random.random())
    torch.manual_seed(0)  # the caller's own state, which must be restored afterwards
    caller_state = torch.get_rng_state()

    randomness = Randomness(SEEDS[1])
    assert torch.equal(randomness.permutation(50), expected[0])  # for example the data split
    with randomness.global_generators():
        assert torch.equal(torch.randperm(20), expected[1])
        np.testing.assert_array_equal(np.random.rand(3), expected[2])
        assert random.random() == expected[3]
    assert torch.equal(torch.get_rng_state(), caller_state)
    torch.manual_seed(SEEDS[1])
    torch.randperm(50), torch.randperm(20)
    assert torch.equal(randomness.permutation(7), torch.randperm(7))  # the run stream moved on


def test_kmeans_is_deterministic() -> None:
    """Repeated k-means runs give the same centers, unlike multi-threaded scikit-learn k-means."""
    from unpaired_rosetta.geometric_initialization import kmeans_centers

    samples = torch.nn.functional.normalize(torch.randn(6000, 64, generator=torch.Generator().manual_seed(0)), dim=1)
    runs = [kmeans_centers(samples, 30, seed=123) for _ in range(4)]
    assert all(torch.equal(run, runs[0]) for run in runs)


def test_the_map_does_not_depend_on_the_callers_thread_count() -> None:
    samples_x, samples_y = clustered_spaces(24_000)
    maps = []
    for num_threads in (1, 8):
        torch.set_num_threads(num_threads)
        aligner = WassersteinProcrustes(num_clusters=10, num_restarts=2, batch_size=4000, num_iterations=2)
        maps.append(aligner.fit(samples_x[:12_000], samples_y[12_000:], randomness=7).weight)
    assert torch.equal(maps[0], maps[1])


def test_mpopt_solves_with_its_own_seed_whatever_ran_before() -> None:
    """A solve does not depend on the previous solve, although pylibmgm reads its seed when the solver is built."""
    generator = torch.Generator().manual_seed(0)
    problems = [
        cka_matching_costs(
            torch.randn(30, 16, generator=generator), torch.randn(30, 24, generator=generator), None, None
        )
        for _ in range(3)
    ]
    solver = MPOptQAPSolver()
    first = [solver.solve(*problem, seed=7).tolist() for problem in problems]
    solver.solve(*problems[0], seed=12345)
    again = [solver.solve(*problem, seed=7).tolist() for problem in problems[::-1]][::-1]
    assert first == again


def test_mpopt_does_not_depend_on_the_global_c_rand_state() -> None:
    """The solver seeds the C library's global rand(), which QPBO inside MPOpt uses to shuffle."""
    generator = torch.Generator().manual_seed(1)
    problem = cka_matching_costs(
        torch.randn(30, 16, generator=generator), torch.randn(30, 24, generator=generator), None, None
    )
    solver = MPOptQAPSolver()
    first = solver.solve(*problem, seed=7).tolist()
    ctypes.CDLL(None).rand()
    assert solver.solve(*problem, seed=7).tolist() == first


def test_floating_point_code_path_is_the_same_on_every_cpu() -> None:
    """MKL and PyTorch use AVX2 (set in the pixi activation env), because AVX-512 kernels round differently."""
    assert os.environ.get("MKL_CBWR") == "AVX2,STRICT"
    assert torch.backends.cpu.get_cpu_capability() == "AVX2"
