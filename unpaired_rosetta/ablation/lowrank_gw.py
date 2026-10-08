"""Cluster matching as an initialization of low-rank Gromov-Wasserstein (Table 2).

A rank-``r`` plan is ``T = Q diag(1/g) R^T`` with non-negative ``Q`` ``[n, r]`` and ``R`` ``[m, r]``, row sums ``a``
and ``b``, and column sums ``g`` (Scetbon, Peyre & Cuturi, 2022). Its Gromov-Wasserstein loss is

    GW(T) = sum_{i,j,k,l} (C_X[i, k] - C_Y[j, l])^2 T[i, j] T[k, l] = a^T C_X^2 a + b^T C_Y^2 b - 2 <C_X T C_Y, T>,

where ``C^2`` squares entrywise.

Table 2 compares five starting plans, each before and after the same mirror descent:

* ``random`` and ``trivial``: the initializations of Scetbon et al. in POT (``_init_lr_sinkhorn``).
* ``lower bound``: low-rank OT between the eccentricities ``(C^2 a)^(1/2)`` (POT's ``lowrank_sinkhorn``).
* ``k-means``: separate k-means on both sets, with cluster ``k`` of X paired with cluster ``k`` of Y.
* ``k-means + QAP``: the same clusters, paired by MPOpt on the standardized kernels of the centers.

The mirror descent is a simplified variant of Algorithm 2 of Scetbon et al. It differs from their code
(``Quad_LGW_MD`` in github.com/meyerscetbon/LinearGromov) as follows:

* The step size is ``gamma = 1 / (0.05 max(|grad_Q|_inf, |grad_R|_inf))``.
* ``g`` takes its own step, is clamped to at least ``1 / (2r)`` and renormalized. ``Q`` and ``R`` are then projected
  separately by 40 Sinkhorn iterations instead of a joint Dykstra projection. The clamp means no component can carry
  less than half of the uniform mass.
* It runs 200 iterations or until the change is below 1e-9, and returns the best iterate projected onto the exact
  marginals.
* The ``k-means`` start uses hard assignments mixed with ``epsilon = 1e-4`` of the uniform plan and a random k-means
  seed.
  The ``lower bound`` start always uses squared Euclidean costs, also for the single-cell data.

These choices reproduce the numbers of the paper. The ranking of the initializations does not depend on them.
"""

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph
from threadpoolctl import threadpool_limits
from torch import Tensor
from torch.nn.functional import one_hot

from unpaired_rosetta.ablation.sinkhorn import sinkhorn_projection
from unpaired_rosetta.geometric_initialization import KMEANS_THREADS, kmeans_centers
from unpaired_rosetta.qap import MPOptQAPSolver
from unpaired_rosetta.randomness import Randomness

INITIALIZATIONS = ["random", "trivial", "k-means", "lower bound", "k-means + QAP"]
EPSILON = 1e-4  # mass of the uniform plan mixed into hard assignments, so every entry is positive
STEP = 0.05  # the mirror descent step is 1 / (STEP * largest gradient entry)
PROJECTION_ITERATIONS = 40  # Sinkhorn iterations per mirror descent step


def uniform(
    num_samples: int,
) -> Tensor:
    return torch.full((num_samples,), 1.0 / num_samples, dtype=torch.float64)


def squared_euclidean_cost(
    samples: Tensor,
) -> Tensor:
    """Squared Euclidean distances, scaled to a maximum of one."""
    cost = torch.cdist(samples, samples) ** 2
    return cost / cost.max()


def geodesic_cost(
    samples: Tensor,
    num_neighbors: int = 50,
) -> Tensor:
    """Shortest paths on the correlation kNN graph (SCOT, Demetci et al., 2020), scaled to a maximum of one.

    Disconnected pairs get the largest finite length.
    """
    graph = kneighbors_graph(
        samples.numpy(), num_neighbors, mode="connectivity", metric="correlation", include_self=True
    )
    distances = dijkstra(graph, directed=False)
    distances[~np.isfinite(distances)] = np.nanmax(distances[np.isfinite(distances)])
    return torch.as_tensor(distances / distances.max(), dtype=torch.float64)


def dense_plan(
    factors: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    q, r, g = factors
    return (q / g.clamp(min=1e-30)[None, :]) @ r.T


def gw_loss(
    cost_x: Tensor,
    cost_y: Tensor,
    plan: Tensor,
) -> float:
    """``sum (C_X[i, k] - C_Y[j, l])^2 T[i, j] T[k, l]``, using the marginals of the plan."""
    a, b = plan.sum(dim=1), plan.sum(dim=0)
    constant = (cost_x**2 @ a) @ a + (cost_y**2 @ b) @ b
    return (constant - 2 * ((cost_x @ plan @ cost_y) * plan).sum()).item()


def plan_foscttm(
    plan: Tensor,
    samples_y: Tensor,
    permutation: Tensor,
) -> float:
    """FOSCTTM of the barycentric projections ``T Y / T 1`` of the samples of X, with strict comparisons."""
    projected = (plan / plan.sum(dim=1, keepdim=True).clamp(min=1e-30)) @ samples_y
    distances = torch.cdist(projected, samples_y)
    true_distances = distances[torch.arange(len(permutation)), permutation][:, None]
    return ((distances < true_distances).double().sum(dim=1) / (len(permutation) - 1)).mean().item()


def kmeans_labels(
    samples: Tensor,
    num_clusters: int,
    seed: int,
) -> Tensor:
    """Index of the nearest k-means center of each sample."""
    return torch.cdist(samples, kmeans_centers(samples, num_clusters, seed)).argmin(dim=1)


def factors_from_clusters(
    labels_x: Tensor,
    labels_y: Tensor,
    rank: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Feasible, strictly positive factors from two clusterings whose clusters ``k`` are paired.

    ``g`` is the mean of the two cluster masses and ``Q = a * ((1 - epsilon) Z_X + epsilon g^T)`` for one-hot
    memberships ``Z_X``, where ``*`` scales row ``i`` by ``a[i]`` (``R`` likewise).
    A Sinkhorn projection then makes the column sums exactly ``g``.
    """
    a, b = uniform(labels_x.shape[0]), uniform(labels_y.shape[0])
    memberships_x = one_hot(labels_x.long(), num_classes=rank).to(a)
    memberships_y = one_hot(labels_y.long(), num_classes=rank).to(b)
    g = (memberships_x.T @ a + memberships_y.T @ b) / 2
    g = g.clamp(min=1e-12)
    g = g / g.sum()
    q = a[:, None] * ((1 - EPSILON) * memberships_x + EPSILON * g[None, :])
    r = b[:, None] * ((1 - EPSILON) * memberships_y + EPSILON * g[None, :])
    return sinkhorn_projection(q.log(), a, g, 2000, 1e-12), sinkhorn_projection(r.log(), b, g, 2000, 1e-12), g


def pot_factors(
    samples_x: Tensor,
    samples_y: Tensor,
    rank: int,
    pot_initialization: str,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """POT's ``random`` or ``deterministic`` (Scetbon et al.'s ``trivial``) factors, in float64."""
    import ot
    from ot.lowrank import _init_lr_sinkhorn

    dtype = torch.float32 if pot_initialization == "deterministic" else torch.float64  # POT mixes in float32 constants
    x, y = samples_x.to(dtype), samples_y.to(dtype)
    a, b = uniform(x.shape[0]).to(dtype), uniform(y.shape[0]).to(dtype)
    factors = _init_lr_sinkhorn(
        x, y, a, b, rank, init=pot_initialization, random_state=seed, reg_init=None, nx=ot.backend.get_backend(x)
    )
    return tuple(factor.double() for factor in factors)


def lower_bound_factors(
    samples_x: Tensor,
    samples_y: Tensor,
    rank: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Low-rank OT between the eccentricities ``(C^2 a)^(1/2)`` of squared Euclidean costs, with exact marginals."""
    import ot

    a, b = uniform(samples_x.shape[0]), uniform(samples_y.shape[0])
    eccentricity_x = torch.sqrt(squared_euclidean_cost(samples_x) ** 2 @ a).reshape(-1, 1)
    eccentricity_y = torch.sqrt(squared_euclidean_cost(samples_y) ** 2 @ b).reshape(-1, 1)
    with threadpool_limits(KMEANS_THREADS):
        q, r, g = ot.lowrank_sinkhorn(
            eccentricity_x,
            eccentricity_y,
            a,
            b,
            rank=rank,
            init="kmeans",
            reg_init=0.1,
            seed_init=seed,
            numItermax=1000,
            stopThr=1e-3,
            warn=False,
        )
    g = g / g.sum()  # POT stops at a loose tolerance
    return sinkhorn_projection(q.log(), a, g, 2000, 1e-12), sinkhorn_projection(r.log(), b, g, 2000, 1e-12), g


def standardized_kernel(
    kernel: Tensor,
) -> Tensor:
    """Off-diagonal entries standardized to mean zero and unit variance, with a zero diagonal."""
    off_diagonal = ~torch.eye(kernel.shape[0], dtype=torch.bool)
    values = kernel[off_diagonal]
    return ((kernel - values.mean()) / values.std()).masked_fill(~off_diagonal, 0.0)


def matched_cluster_factors(
    samples_x: Tensor,
    samples_y: Tensor,
    rank: int,
    randomness: Randomness,
) -> tuple[Tensor, Tensor, Tensor]:
    """k-means on both sets, with clusters paired by MPOpt on ``max_P Tr(K_A P K_B P^T)``.

    ``K_A`` and ``K_B`` are the standardized kernels of the centers, as in the runs behind Table 2. Our geometric
    initialization uses centered kernels instead.
    """
    seed_x, seed_y = randomness.kmeans_seed(), randomness.kmeans_seed()
    centers_x = kmeans_centers(samples_x, rank, seed_x)
    centers_y = kmeans_centers(samples_y, rank, seed_y)
    labels_x = torch.cdist(samples_x, centers_x).argmin(dim=1)
    labels_y = torch.cdist(samples_y, centers_y).argmin(dim=1)
    kernel_x, kernel_y = standardized_kernel(centers_x @ centers_x.T), standardized_kernel(centers_y @ centers_y.T)
    permutation = MPOptQAPSolver().solve(-kernel_x, kernel_y, None, randomness.solver_seed())
    partner_of_y_cluster = torch.empty_like(permutation)
    partner_of_y_cluster[permutation] = torch.arange(rank)
    return factors_from_clusters(labels_x, partner_of_y_cluster[labels_y], rank)


def initial_factors(
    samples_x: Tensor,
    samples_y: Tensor,
    rank: int,
    initialization: str,
    randomness: Randomness,
) -> tuple[Tensor, Tensor, Tensor]:
    """Starting plan ``(Q, R, g)`` for one of the ``INITIALIZATIONS``."""
    if initialization == "k-means + QAP":
        return matched_cluster_factors(samples_x, samples_y, rank, randomness)
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=randomness.torch).item())
    if initialization == "random":
        return pot_factors(samples_x, samples_y, rank, "random", seed)
    if initialization == "trivial":
        return pot_factors(samples_x, samples_y, rank, "deterministic", seed)
    if initialization == "lower bound":
        return lower_bound_factors(samples_x, samples_y, rank, seed)
    if initialization == "k-means":
        labels_x = kmeans_labels(samples_x, rank, randomness.kmeans_seed())
        labels_y = kmeans_labels(samples_y, rank, randomness.kmeans_seed())
        return factors_from_clusters(labels_x, labels_y, rank)
    raise ValueError(f"Unknown initialization {initialization!r}; expected one of {INITIALIZATIONS}.")


def mirror_descent(
    factors: tuple[Tensor, Tensor, Tensor],
    cost_x: Tensor,
    cost_y: Tensor,
    num_iterations: int = 200,
    tolerance: float = 1e-9,
) -> tuple[tuple[Tensor, Tensor, Tensor], list[float]]:
    """Maximize ``<C_X T C_Y, T>``, which minimizes the GW loss, over the factors by KL mirror descent.

    Returns the best iterate projected onto the exact marginals, and the objective of each iterate. The first value
    is for the start and the last for the returned factors.
    """
    q, r, g = (factor.double() for factor in factors)
    rank = g.shape[0]
    a, b = uniform(q.shape[0]), uniform(r.shape[0])
    history = []
    best = (q, r, g, -float("inf"))
    for _ in range(num_iterations):
        inverse_g = 1.0 / g.clamp(min=1e-30)
        plan = (q * inverse_g[None, :]) @ r.T
        gradient = cost_x @ plan @ cost_y  # gradient of <C_X T C_Y, T> in T, up to a factor 2
        value = (gradient * plan).sum().item()
        gradient_times_r = gradient @ r
        gradient_q = 2.0 * gradient_times_r * inverse_g[None, :]
        gradient_r = 2.0 * (gradient.T @ q) * inverse_g[None, :]
        gradient_g = (q * gradient_times_r).sum(dim=0) * inverse_g  # -g * (gradient in g) / 2, entrywise
        history.append(value)
        if value > best[3]:
            best = (q, r, g, value)

        largest = max(gradient_q.abs().max().item(), gradient_r.abs().max().item(), 1e-30)
        step = 1.0 / (largest * STEP)
        new_g = torch.softmax(g.clamp(min=1e-30).log() - step * gradient_g / g.clamp(min=1e-30), dim=0)
        new_g = new_g.clamp(min=1.0 / (2 * rank))
        new_g = new_g / new_g.sum()
        new_q = sinkhorn_projection(q.clamp(min=1e-30).log() + step * gradient_q, a, new_g, PROJECTION_ITERATIONS)
        new_r = sinkhorn_projection(r.clamp(min=1e-30).log() + step * gradient_r, b, new_g, PROJECTION_ITERATIONS)

        change = max((new_q - q).abs().max().item(), (new_r - r).abs().max().item())
        q, r, g = new_q, new_r, new_g
        if change < tolerance:
            break

    q, r, g, _ = best
    q = sinkhorn_projection(q.clamp(min=1e-30).log(), a, g, 2000, 1e-12)
    r = sinkhorn_projection(r.clamp(min=1e-30).log(), b, g, 2000, 1e-12)
    plan = dense_plan((q, r, g))
    history.append(((cost_x @ plan @ cost_y) * plan).sum().item())
    return (q, r, g), history
