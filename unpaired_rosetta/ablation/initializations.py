"""Other estimates of the coarse correspondence ``M = X^T T Y`` (Table 1).

Each returns the cross-covariance ``M`` that our read-out turns into a first map. Like ours, each averages
``S = 30`` restarts on random subsets.

* ``PCAHeuristic``: pair the leading principal directions of both spaces (Hoshen & Wolf, 2018).
* ``SortedHeuristic``: mutual nearest neighbors of the row-sorted square roots of the Gram matrices (Artetxe et
  al., 2018).
* ``GromovWasserstein``: POT's Gromov-Wasserstein solver on 1024 samples, started from the uniform or a random plan
  (Peyre et al., 2016).
* ``ClusterMatching``: our clustering and matching with MPOpt or with mini-vec2vec's 2-opt (Dar, 2025). Unlike
  ``GeometricInitialization`` it returns the matched centers, which some ablations need.

Each restart first draws the subsets of both sides and then the draws of the initialization, as in the code that
produced the paper's numbers.
"""

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from unpaired_rosetta.ablation.sinkhorn import sinkhorn_projection
from unpaired_rosetta.assignment import batched_hungarian_matching
from unpaired_rosetta.baselines.mini_vec2vec import two_opt
from unpaired_rosetta.determinism import FIT_THREADS
from unpaired_rosetta.geometric_initialization import cka_matching_costs, kmeans_centers
from unpaired_rosetta.qap import MPOptQAPSolver
from unpaired_rosetta.randomness import Randomness

HEURISTIC_SAMPLES = 1024  # subset size for the two heuristics and Gromov-Wasserstein
NUM_RESTARTS = 30  # S, as for our initialization
TWO_OPT_REPETITIONS = 30  # mini-vec2vec keeps the best of 30 random 2-opt searches


def random_subset(
    samples: Tensor,
    size: int,
    randomness: Randomness,
) -> Tensor:
    """``size`` random rows, or all rows in order if there are not more."""
    if size >= samples.shape[0]:
        return samples
    return samples[randomness.subset(samples.shape[0], size)]


def matched_cross_covariance(
    matched_x: Tensor,
    matched_y: Tensor,
) -> Tensor:
    """``X^T T Y`` for the plan with mass ``1/k`` on each of ``k`` matched pairs."""
    weights = torch.full((matched_x.shape[0],), 1.0 / max(matched_x.shape[0], 1), dtype=matched_x.dtype)
    return (matched_x * weights[:, None]).T @ matched_y


class PCAHeuristic:
    """Map the top ``50`` principal directions of X onto those of Y and match samples under that map.

    Principal directions have no sign, so each is flipped to give its projections a positive third moment.
    """

    num_components = 50

    def __call__(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        randomness: Randomness,
    ) -> Tensor:
        cross_covariances = []
        for _ in range(NUM_RESTARTS):
            batch_x = random_subset(samples_x, HEURISTIC_SAMPLES, randomness)
            batch_y = random_subset(samples_y, HEURISTIC_SAMPLES, randomness)
            cross_covariances.append(self.restart(batch_x, batch_y))
        return torch.stack(cross_covariances).mean(dim=0)

    def restart(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
    ) -> Tensor:
        rank = min(self.num_components, samples_x.shape[1], samples_y.shape[1])
        weight = principal_directions(samples_x, rank).T @ principal_directions(samples_y, rank)
        block_size = max(samples_x.shape[0], samples_y.shape[0])
        rows, columns = batched_hungarian_matching(samples_x @ weight, samples_y, block_size)
        return matched_cross_covariance(samples_x[rows], samples_y[columns])


def principal_directions(
    samples: Tensor,
    rank: int,
) -> Tensor:
    """The ``rank`` leading right singular vectors of the centered samples, signed by the skewness of the projections.

    The skewness uses the samples as given, which the preprocessing has already centered.
    """
    _, _, right = torch.linalg.svd(samples - samples.mean(dim=0), full_matrices=False)
    right = right[:rank]
    skewness = ((samples @ right.T) ** 3).mean(dim=0)
    return right * torch.where(skewness < 0, -1.0, 1.0)[:, None]


class SortedHeuristic:
    """Sort each row of ``(X X^T)^(1/2)`` and ``(Y Y^T)^(1/2)`` and keep mutual nearest neighbors of the sorted rows.

    Sorting removes the dependence on the sample order, so rows of the two spaces can be compared.
    """

    def __call__(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        randomness: Randomness,
    ) -> Tensor:
        cross_covariances = []
        for _ in range(NUM_RESTARTS):
            batch_x = random_subset(samples_x, HEURISTIC_SAMPLES, randomness)
            batch_y = random_subset(samples_y, HEURISTIC_SAMPLES, randomness)
            cross_covariances.append(self.restart(batch_x, batch_y))
        return torch.stack(cross_covariances).mean(dim=0)

    @staticmethod
    def restart(
        samples_x: Tensor,
        samples_y: Tensor,
    ) -> Tensor:
        size = min(samples_x.shape[0], samples_y.shape[0])  # sorted rows must have equal length
        samples_x, samples_y = samples_x[:size], samples_y[:size]
        distances = torch.cdist(sorted_gram_root(samples_x), sorted_gram_root(samples_y))
        nearest_y, nearest_x = distances.argmin(dim=1), distances.argmin(dim=0)
        rows = (nearest_x[nearest_y] == torch.arange(size)).nonzero(as_tuple=True)[0]
        return matched_cross_covariance(samples_x[rows], samples_y[nearest_y[rows]])


def sorted_gram_root(
    samples: Tensor,
) -> Tensor:
    """``(X X^T)^(1/2) = U S^(1/2) U^T`` for ``X = U S V^T``, with each row sorted in ascending order."""
    left, singular_values, _ = torch.linalg.svd(samples, full_matrices=False)
    root = (left * singular_values.clamp(min=0).sqrt()) @ left.T
    return root.sort(dim=1).values


class GromovWasserstein:
    """POT's Gromov-Wasserstein solver on the linear kernels of 1024 samples per side.

    Uses the square loss, uniform masses and at most 1000 iterations. ``initial_plan`` is ``"uniform"`` or
    ``"random"`` (uniform random entries projected onto the couplings).
    """

    max_iterations = 1000

    def __init__(
        self,
        initial_plan: str,
    ) -> None:
        if initial_plan not in ("uniform", "random"):
            raise ValueError(f"Unknown initial plan {initial_plan!r}")
        self.initial_plan = initial_plan

    def __call__(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        randomness: Randomness,
    ) -> Tensor:
        cross_covariances = []
        for _ in range(NUM_RESTARTS):
            batch_x = random_subset(samples_x, HEURISTIC_SAMPLES, randomness)
            batch_y = random_subset(samples_y, HEURISTIC_SAMPLES, randomness)
            cross_covariances.append(self.restart(batch_x, batch_y, randomness))
        return torch.stack(cross_covariances).mean(dim=0)

    def restart(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        randomness: Randomness,
    ) -> Tensor:
        import ot

        x, y = samples_x.double(), samples_y.double()
        masses_x = torch.full((x.shape[0],), 1.0 / x.shape[0], dtype=torch.float64)
        masses_y = torch.full((y.shape[0],), 1.0 / y.shape[0], dtype=torch.float64)
        if self.initial_plan == "uniform":
            plan = torch.outer(masses_x, masses_y)
        else:
            log_kernel = torch.rand(x.shape[0], y.shape[0], dtype=torch.float64, generator=randomness.torch).log()
            plan = sinkhorn_projection(log_kernel, masses_x, masses_y, num_iterations=1000)
        plan = ot.gromov.gromov_wasserstein(
            x @ x.T,
            y @ y.T,
            p=masses_x,
            q=masses_y,
            loss_fun="square_loss",
            symmetric=True,
            G0=plan,
            max_iter=self.max_iterations,
        )
        return (x.T @ plan @ y).to(samples_x.dtype)


@dataclass
class ClusterRestart:
    """Random choices of one restart: the rows of both batches (None means all rows) and the seeds.

    ``solver_seeds`` holds one seed for MPOpt or one per 2-opt search.
    """

    rows_x: Tensor | None
    rows_y: Tensor | None
    kmeans_seed_x: int
    kmeans_seed_y: int
    solver_seeds: list[int]


class ClusterMatching:
    """Algorithm 2 of the paper with a choice of QAP solver, returning the matched centers of each restart.

    With ``solver="MPOpt"``, ``average_cross_covariance`` of the result equals the ``M`` of
    ``GeometricInitialization`` bit for bit.
    """

    def __init__(
        self,
        num_clusters: int = 30,
        solver: str = "MPOpt",
        num_restarts: int = NUM_RESTARTS,
        batch_size: int = 10_000,
        num_workers: int = 1,
    ) -> None:
        if solver not in ("MPOpt", "2-opt"):
            raise ValueError(f"Unknown QAP solver {solver!r}")
        self.num_clusters = num_clusters
        self.solver = solver
        self.num_restarts = num_restarts
        self.batch_size = batch_size
        self.num_workers = num_workers  # parallel processes do not change the result

    def __call__(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        randomness: Randomness,
    ) -> list[tuple[Tensor, Tensor]]:
        """``(centers_x, matched_centers_y)`` for each restart, row-aligned."""
        restarts = [
            self.draw_restart(samples_x.shape[0], samples_y.shape[0], randomness) for _ in range(self.num_restarts)
        ]
        arguments = [(restart, self.num_clusters, self.solver) for restart in restarts]
        return map_in_processes(match_clusters, arguments, (samples_x, samples_y), self.num_workers)

    def draw_restart(
        self,
        num_samples_x: int,
        num_samples_y: int,
        randomness: Randomness,
    ) -> ClusterRestart:
        rows_x = None if num_samples_x <= self.batch_size else randomness.subset(num_samples_x, self.batch_size)
        rows_y = None if num_samples_y <= self.batch_size else randomness.subset(num_samples_y, self.batch_size)
        kmeans_seed_x, kmeans_seed_y = randomness.kmeans_seed(), randomness.kmeans_seed()
        num_seeds = 1 if self.solver == "MPOpt" else TWO_OPT_REPETITIONS
        solver_seeds = [randomness.solver_seed() for _ in range(num_seeds)]
        return ClusterRestart(rows_x, rows_y, kmeans_seed_x, kmeans_seed_y, solver_seeds)


def match_clusters(
    restart: ClusterRestart,
    num_clusters: int,
    solver: str,
    samples_x: Tensor,
    samples_y: Tensor,
) -> tuple[Tensor, Tensor]:
    """Cluster both batches and permute the centers of Y to maximize the linear CKA of the centers."""
    batch_x = samples_x if restart.rows_x is None else samples_x[restart.rows_x]
    batch_y = samples_y if restart.rows_y is None else samples_y[restart.rows_y]
    centers_x = kmeans_centers(batch_x, num_clusters, restart.kmeans_seed_x)
    centers_y = kmeans_centers(batch_y, num_clusters, restart.kmeans_seed_y)
    cost_x, cost_y, _ = cka_matching_costs(centers_x, centers_y, None, None)
    if solver == "MPOpt":
        permutation = MPOptQAPSolver().solve(cost_x, cost_y, None, restart.solver_seeds[0])
    else:
        permutation = best_two_opt(cost_x, cost_y, restart.solver_seeds)
    return centers_x, centers_y[permutation]


def average_cross_covariance(
    matched_centers: list[tuple[Tensor, Tensor]],
) -> Tensor:
    """``M = 1/(SC) sum_s A_s^T P_s B_s``, computed exactly as in ``GeometricInitialization``."""
    return torch.stack(
        [matched_cross_covariance(centers_x, centers_y) for centers_x, centers_y in matched_centers]
    ).mean(dim=0)


def best_two_opt(
    cost_x: Tensor,
    cost_y: Tensor,
    seeds: list[int],
) -> Tensor:
    """Best of several 2-opt searches for ``min_pi sum_{i,k} cost_x[i, k] cost_y[pi(i), pi(k)]``, one per seed.

    Ties go to the earlier seed.
    """
    kernel_x, kernel_y = (-cost_x).double().numpy(), cost_y.double().numpy()
    best_score, best_permutation = -np.inf, None
    for seed in seeds:
        start = np.random.RandomState(seed).permutation(kernel_x.shape[0])
        permutation, score = two_opt(kernel_x, kernel_y, start)
        if score > best_score:
            best_score, best_permutation = score, permutation
    return torch.as_tensor(best_permutation, dtype=torch.long)


_worker_data: tuple[Tensor, ...] = ()


def _store_worker_data(
    *data: Tensor,
) -> None:
    global _worker_data
    _worker_data = data
    torch.set_num_threads(FIT_THREADS)  # as in the main process, so parallel and sequential restarts agree


def _call_in_worker(
    function: Callable[..., object],
    arguments: tuple,
) -> object:
    return function(*arguments, *_worker_data)


def map_in_processes(
    function: Callable[..., object],
    arguments: list[tuple],
    data: tuple[Tensor, ...],
    num_workers: int,
) -> list:
    """``[function(*a, *data) for a in arguments]`` in ``num_workers`` spawned processes that receive ``data`` once.

    Processes, not threads, because MPOpt keeps its seed in a class attribute.
    """
    progress = dict(desc="cluster matching", leave=False)
    if num_workers == 1:
        return [function(*argument, *data) for argument in tqdm(arguments, **progress)]
    context = get_context("spawn")
    with ProcessPoolExecutor(
        num_workers, mp_context=context, initializer=_store_worker_data, initargs=data
    ) as executor:
        futures = [executor.submit(_call_in_worker, function, argument) for argument in arguments]
        return [future.result() for future in tqdm(futures, **progress)]
