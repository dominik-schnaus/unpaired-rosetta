"""Geometric initialization (Algorithm 2 of the paper).

For each of ``S`` restarts, draw ``b`` samples of each space, cluster them into ``C`` k-means centers, and match the
centers by the permutation that maximizes their linear CKA (a quadratic assignment problem). The average ``M`` of
the cross-covariances ``(1/C) A_s^T P_s B_s`` is a coarse correspondence between the two spaces.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from torch import Tensor
from tqdm.auto import tqdm

from unpaired_rosetta.determinism import FIT_THREADS, KMEANS_THREADS
from unpaired_rosetta.qap import MPOptQAPSolver
from unpaired_rosetta.randomness import Randomness


@dataclass
class Restart:
    """Random choices of one restart: the rows of both batches (None means all rows) and the seeds."""

    rows_x: Tensor | None
    rows_y: Tensor | None
    kmeans_seed_x: int
    kmeans_seed_y: int
    solver_seed: int


def kmeans_centers(
    samples: Tensor,
    num_clusters: int,
    seed: int,
) -> Tensor:
    """Centers of scikit-learn's k-means with default parameters."""
    with threadpool_limits(KMEANS_THREADS):
        model = KMeans(n_clusters=num_clusters, init="k-means++", random_state=seed).fit(samples.numpy())
    return torch.as_tensor(model.cluster_centers_, dtype=samples.dtype)


def centered_kernel(
    samples: Tensor,
) -> Tensor:
    """``H K H`` for the linear kernel ``K = samples samples^T`` and the centering matrix ``H = I - 11^T/n``."""
    gram = samples @ samples.T
    return gram - gram.mean(dim=0, keepdim=True) - gram.mean(dim=1, keepdim=True) + gram.mean()


def cka_matching_costs(
    centers_x: Tensor,
    centers_y: Tensor,
    paired_x: Tensor | None,
    paired_y: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """QAP costs whose minimizing permutation maximizes ``CKA(vstack(A, X_hat), vstack(P B, Y_hat))`` (Lemmas 1 and 3 of the paper).

    Without pairs, ``argmax_P CKA(A, P B) = argmax_P Tr(K_A P K_B P^T)`` for the centered kernels ``K_A`` and ``K_B``.
    With pairs, the centers and pairs are centered together. The pair rows are fixed, so they only add the linear term
    ``2 Tr(K_12 L_12^T P^T)``, where ``K_12`` and ``L_12`` are the center-pair blocks of the jointly centered kernels.

    Returns:
        The quadratic costs of X and Y and the linear cost (None without pairs) for ``MPOptQAPSolver.solve``.
    """
    if paired_x is None:
        return -centered_kernel(centers_x), centered_kernel(centers_y), None
    kernel_x, cross_kernel_x = jointly_centered_kernels(centers_x, paired_x)
    kernel_y, cross_kernel_y = jointly_centered_kernels(centers_y, paired_y)
    linear = (-2.0 * cross_kernel_x) @ cross_kernel_y.T
    return -kernel_x, kernel_y, linear


def jointly_centered_kernels(
    centers: Tensor,
    paired: Tensor,
) -> tuple[Tensor, Tensor]:
    """Blocks ``K_11`` (center-center) and ``K_12`` (center-pair) of the centered kernel of ``vstack(centers, paired)``."""
    row_weight = 1.0 / (centers.shape[0] + paired.shape[0])
    mean = row_weight * centers.sum(dim=0) + row_weight * paired.sum(dim=0)
    centered_centers = centers - mean
    centered_paired = paired - mean
    return centered_centers @ centered_centers.T, centered_centers @ centered_paired.T


def solve_restart(
    restart: Restart,
    samples_x: Tensor,
    samples_y: Tensor,
    paired_x: Tensor | None,
    paired_y: Tensor | None,
    num_clusters: int,
) -> Tensor:
    """Cross-covariance ``(1/C) A_s^T P_s B_s`` of one matched clustering (lines 5-7 of Algorithm 2)."""
    batch_x = samples_x if restart.rows_x is None else samples_x[restart.rows_x]
    batch_y = samples_y if restart.rows_y is None else samples_y[restart.rows_y]
    centers_x = kmeans_centers(batch_x, num_clusters, restart.kmeans_seed_x)
    centers_y = kmeans_centers(batch_y, num_clusters, restart.kmeans_seed_y)
    cost_x, cost_y, linear = cka_matching_costs(centers_x, centers_y, paired_x, paired_y)
    permutation = MPOptQAPSolver().solve(cost_x, cost_y, linear, restart.solver_seed)
    uniform_weights = torch.full((num_clusters,), 1.0 / num_clusters, dtype=centers_x.dtype)
    return (centers_x * uniform_weights[:, None]).T @ centers_y[permutation]


class GeometricInitialization:
    """Coarse correspondence ``M = 1/(SC) sum_s A_s^T P_s B_s`` from ``S`` matched clusterings.

    Args:
        num_clusters: Number of k-means centers ``C`` per restart.
        num_restarts: Number of restarts ``S``.
        batch_size: Rows ``b`` drawn from each space per restart.
        num_workers: Parallel processes. They do not change the result. They are spawned, so a script with
            ``num_workers > 1`` needs an ``if __name__ == "__main__":`` guard.
    """

    def __init__(
        self,
        num_clusters: int = 30,
        num_restarts: int = 30,
        batch_size: int = 10_000,
        num_workers: int = 1,
    ) -> None:
        self.num_clusters = num_clusters
        self.num_restarts = num_restarts
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __call__(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None,
        paired_y: Tensor | None,
        randomness: Randomness,
    ) -> Tensor:
        """Average cross-covariance ``M`` ``[d_X, d_Y]`` of the preprocessed samples and optional known pairs."""
        restarts = [
            self.draw_restart(samples_x.shape[0], samples_y.shape[0], randomness) for _ in range(self.num_restarts)
        ]
        data = (samples_x, samples_y, paired_x, paired_y)
        if self.num_workers == 1:
            cross_covariances = [
                solve_restart(restart, *data, self.num_clusters) for restart in tqdm(restarts, **PROGRESS)
            ]
        else:
            cross_covariances = self.solve_restarts_in_parallel(restarts, data)
        return torch.stack(cross_covariances).mean(dim=0)

    def draw_restart(
        self,
        num_samples_x: int,
        num_samples_y: int,
        randomness: Randomness,
    ) -> Restart:
        """Random batches of ``b`` rows of both spaces (all rows if a space has fewer) and the seeds."""
        rows_x = None if num_samples_x <= self.batch_size else randomness.subset(num_samples_x, self.batch_size)
        rows_y = None if num_samples_y <= self.batch_size else randomness.subset(num_samples_y, self.batch_size)
        return Restart(rows_x, rows_y, randomness.kmeans_seed(), randomness.kmeans_seed(), randomness.solver_seed())

    def solve_restarts_in_parallel(
        self,
        restarts: list[Restart],
        data: tuple[Tensor, Tensor, Tensor | None, Tensor | None],
    ) -> list[Tensor]:
        """Solve the restarts in worker processes that each receive the data once.

        Processes, not threads, because MPOpt keeps its seed in a class attribute.
        """
        context = get_context("spawn")
        with ProcessPoolExecutor(
            self.num_workers, mp_context=context, initializer=_store_worker_data, initargs=data
        ) as executor:
            futures = [executor.submit(_solve_restart_in_worker, restart, self.num_clusters) for restart in restarts]
            return [future.result() for future in tqdm(futures, **PROGRESS)]


PROGRESS = dict(desc="geometric initialization", leave=False)
_worker_data: tuple[Tensor | None, ...] = ()


def _store_worker_data(
    *data: Tensor | None,
) -> None:
    global _worker_data
    _worker_data = data
    torch.set_num_threads(FIT_THREADS)  # as in the main process, so parallel and sequential restarts agree


def _solve_restart_in_worker(
    restart: Restart,
    num_clusters: int,
) -> Tensor:
    return solve_restart(restart, *_worker_data, num_clusters)
