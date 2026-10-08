"""SCOT+ (Baker et al., 2026) turned into a map between the two spaces, the single-cell baseline on PBMC.

SCOT+ solves entropic Gromov-Wasserstein on shortest-path distances in kNN graphs. The number of neighbors ``k`` and
the entropy ``epsilon`` are chosen without labels by SCOT's self-tuning, which keeps the coupling with the lowest
Gromov-Wasserstein cost over a grid.

SCOT only aligns the cells it was fitted on, but our protocol evaluates on unseen samples. So each coupling is turned
into an orthogonal map by projecting X onto Y with the coupling and fitting orthogonal Procrustes. Couplings are
solved on random subsets of 1000 samples per side. Five such maps are averaged with softmax weights on a label-free
fit score and projected back to an orthogonal map.
"""

import numpy as np
import torch
from scotplus.solvers import SinkhornSolver
from scotplus.utils.alignment import compute_graph_distances
from torch import Tensor
from torch.nn.functional import normalize

from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.linalg import polar
from unpaired_rosetta.randomness import Randomness

# SCOT's self-tuning grid (``scotv1.SCOT.unsupervised_scot`` for more than 250 samples).
NUM_NEIGHBORS_GRID = (20, 40, 60, 80, 100)
EPSILON_GRID = tuple(float(epsilon) for epsilon in np.logspace(-1, -3, 12))
SCORE_SIZE = 1000  # training samples per side used to score the ensemble members
# Small epsilon needs this many iterations to converge.
NUM_BCD_ITERATIONS = 20
NUM_SINKHORN_ITERATIONS = 20_000


def graph_distances(
    samples: Tensor,
    num_neighbors: int,
) -> Tensor:
    """Shortest-path distances in the kNN graph (correlation metric), scaled to a maximum of 1."""
    distances = compute_graph_distances(
        samples.cpu().numpy(), n_neighbors=num_neighbors, mode="connectivity", metric="correlation"
    )
    return torch.as_tensor(distances, dtype=torch.float64, device=samples.device)


def gromov_wasserstein_cost(
    distances_x: Tensor,
    distances_y: Tensor,
    coupling: Tensor,
) -> float:
    """``<|D_X - D_Y|^2, P (x) P>`` (``(x)`` is the tensor product), the criterion of the self-tuning."""
    marginal_x, marginal_y = coupling.sum(1), coupling.sum(0)
    constant = (distances_x**2 @ marginal_x)[:, None] + (distances_y**2 @ marginal_y)[None, :]
    return float((coupling * (constant - 2 * distances_x @ coupling @ distances_y.T)).sum())


def self_tuned_coupling(
    subset_x: Tensor,
    subset_y: Tensor,
    solver: SinkhornSolver,
) -> Tensor:
    """The entropic Gromov-Wasserstein coupling with the lowest cost over the (k, epsilon) grid.

    Couplings that are not finite or lost more than half of their mass did not converge and are skipped.
    """
    best_cost, best_coupling = None, None
    for num_neighbors in NUM_NEIGHBORS_GRID:
        distances_x, distances_y = graph_distances(subset_x, num_neighbors), graph_distances(subset_y, num_neighbors)
        for epsilon in EPSILON_GRID:
            coupling = solver.gw(distances_x, distances_y, eps=epsilon, verbose=False)[0].to(torch.float64)
            if not torch.isfinite(coupling).all() or coupling.sum() < 0.5:
                continue
            cost = gromov_wasserstein_cost(distances_x, distances_y, coupling)
            if best_cost is None or cost < best_cost:
                best_cost, best_coupling = cost, coupling
    if best_coupling is None:
        raise RuntimeError("No setting of the self-tuning grid produced a usable coupling.")
    return best_coupling


def fit_score(
    scores_x: Tensor,
    scores_y: Tensor,
    weight: Tensor,
) -> float:
    """Mean cosine of each mapped sample of X to its nearest sample of Y. Higher is better."""
    mapped = normalize(scores_x @ weight, dim=-1)
    return (mapped @ normalize(scores_y, dim=-1).T).max(dim=1).values.mean().item()


def softmax_average(
    maps: list[Tensor],
    scores: list[float],
) -> Tensor:
    """Average of the member maps with softmax weights on their scores, projected to an orthogonal map."""
    stacked = torch.stack(maps, dim=0)
    scores = torch.tensor(scores, dtype=stacked.dtype, device=stacked.device)
    weights = torch.softmax((scores - scores.max()) / (scores.std() + 1e-8), dim=0)
    return polar((weights[:, None, None] * stacked).sum(dim=0))


class SCOTPlus:
    """SCOT+ map in float64 on ``device``. The paper's runs used a GPU."""

    def __init__(
        self,
        num_members: int = 5,
        subset_size: int = 1000,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_members = num_members
        self.subset_size = subset_size
        self.device = torch.device(device)
        self.solver = SinkhornSolver(
            nits_bcd=NUM_BCD_ITERATIONS, nits_uot=NUM_SINKHORN_ITERATIONS, tol_uot=1e-5, device=self.device
        )
        self.weight = None

    def subset(
        self,
        samples: Tensor,
    ) -> Tensor:
        """A random subset of ``subset_size`` rows, drawn from the global generator of ``device``."""
        if samples.shape[0] <= self.subset_size:
            return samples
        return samples[torch.randperm(samples.shape[0], device=samples.device)[: self.subset_size]]

    def member_map(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
    ) -> Tensor:
        """Couple random subsets of both sides, project X onto Y with the coupling, and fit orthogonal Procrustes."""
        subset_x, subset_y = self.subset(samples_x), self.subset(samples_y)
        coupling = self_tuned_coupling(subset_x, subset_y, self.solver)
        projected = normalize(coupling @ subset_y, dim=-1)
        return polar(
            polar(subset_x.T @ projected)
        )  # the second polar only changes rounding, kept to match the paper's runs

    @torch.inference_mode()
    @torch_threads()
    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None = None,
        paired_y: Tensor | None = None,
        randomness: Randomness | int = 0,
    ) -> "SCOTPlus":
        """Fit on two unpaired sets. Known pairs are ignored."""
        if not isinstance(randomness, Randomness):
            randomness = Randomness(randomness)
        samples_x = samples_x.to(self.device, torch.float64)
        samples_y = samples_y.to(self.device, torch.float64)
        maps, scores = [], []
        with randomness.global_generators():
            for _ in range(self.num_members):
                maps.append(self.member_map(samples_x, samples_y))
                scores.append(fit_score(samples_x[:SCORE_SIZE], samples_y[:SCORE_SIZE], maps[-1]))
        self.weight = softmax_average(maps, scores)
        return self

    @torch.inference_mode()
    def similarity(
        self,
        queries_x: Tensor,
        keys_y: Tensor,
    ) -> Tensor:
        """Inner products ``x W y^T`` of the unit-length embeddings."""
        queries_x = queries_x.to(self.device, torch.float64)
        keys_y = keys_y.to(self.device, torch.float64)
        return (queries_x @ self.weight) @ keys_y.T
