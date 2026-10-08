"""mini-vec2vec (Dar, 2025): unpaired alignment by matched clusters, relative representations and refinement.

Follows the official notebook (github.com/guy-dar/mini-vec2vec, ``linear_vec2vec.ipynb``):

1. Anchors: in each of 30 runs, cluster 10,000 random samples of each space into 20 centers and match them by the
   best of 30 random 2-opt searches for ``max_P Tr(K_A P K_B P^T)``.
2. Read-out: describe each sample by its cosine similarities to the anchors, pair each sample of X with the mean of
   its 50 most similar samples of Y, and solve Procrustes.
3. refine1: 100 times, pair 1,000 random samples ``x`` with the mean of the 50 samples of Y closest to ``x W``, and
   move ``W`` halfway to the Procrustes solution.
4. refine2: cluster X into 500 centers, cluster Y starting from the mapped centers, and move ``W`` halfway to the
   Procrustes solution of the centers.

As in the notebook, ``W`` is an average of orthogonal maps and is not projected back. The random draws come in the
order that produced the NQ numbers of the paper.
"""

import itertools

import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from torch import Tensor
from torch.nn.functional import embedding_bag, normalize
from tqdm.auto import trange

from unpaired_rosetta.determinism import KMEANS_THREADS, torch_threads
from unpaired_rosetta.geometric_initialization import centered_kernel
from unpaired_rosetta.linalg import center_and_normalize, polar
from unpaired_rosetta.randomness import Randomness


def kmeans(
    samples: Tensor,
    num_clusters: int,
    seed: int,
    init: Tensor | None = None,
) -> Tensor:
    """Centers of scikit-learn's k-means, started with k-means++ or from the centers ``init``."""
    init = "k-means++" if init is None else init.numpy()
    with threadpool_limits(KMEANS_THREADS):
        model = KMeans(n_clusters=num_clusters, init=init, random_state=seed).fit(samples.numpy())
    return torch.as_tensor(model.cluster_centers_, dtype=samples.dtype)


def two_opt(
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    permutation: np.ndarray,
) -> tuple[np.ndarray, float]:
    """2-opt local search for ``max_P Tr(K_X P K_Y P^T)`` starting from ``permutation``.

    Takes the first improving swap until none is left, like ``scipy.optimize.quadratic_assignment(method="2opt")``.
    """
    permutation = permutation.copy()
    best_score = np.sum(kernel_x * kernel_y[permutation][:, permutation])
    improved = True
    while improved:
        improved = False
        for i, j in itertools.combinations(range(len(permutation)), 2):
            permutation[i], permutation[j] = permutation[j], permutation[i]
            score = np.sum(kernel_x * kernel_y[permutation][:, permutation])
            if score > best_score:
                best_score = score
                improved = True
                break
            permutation[i], permutation[j] = permutation[j], permutation[i]
    return permutation, best_score


def match_clusters(
    centers_x: Tensor,
    centers_y: Tensor,
    num_restarts: int,
    randomness: Randomness,
) -> Tensor:
    """Permutation of ``centers_y`` that best matches the centered kernels, from ``num_restarts`` random 2-opt runs."""
    kernel_x = centered_kernel(centers_x).double().numpy()
    kernel_y = centered_kernel(centers_y).double().numpy()
    best_score, best_permutation = -np.inf, None
    for _ in range(num_restarts):
        start = np.random.RandomState(randomness.solver_seed()).permutation(len(kernel_x))
        permutation, score = two_opt(kernel_x, kernel_y, start)
        if score > best_score:
            best_score, best_permutation = score, permutation
    return torch.as_tensor(best_permutation, dtype=torch.long)


def mean_of_neighbors(
    indices: Tensor,
    targets: Tensor,
) -> Tensor:
    """Row ``i`` is the mean of ``targets[indices[i]]``."""
    offsets = torch.arange(0, indices.numel(), indices.shape[1])
    return embedding_bag(indices.flatten(), targets, offsets=offsets, mode="mean")


def cosine_neighbors(
    queries: Tensor,
    keys: Tensor,
    num_neighbors: int,
    block_size: int = 4096,
) -> Tensor:
    """Indices of the ``num_neighbors`` keys with the largest cosine similarity to each query."""
    queries, keys = normalize(queries, dim=-1), normalize(keys, dim=-1)
    indices = torch.empty(queries.shape[0], num_neighbors, dtype=torch.long)
    for start in range(0, queries.shape[0], block_size):
        indices[start : start + block_size] = (queries[start : start + block_size] @ keys.T).topk(num_neighbors).indices
    return indices


class MiniVec2Vec:
    def __init__(
        self,
        num_clusters: int = 20,
        num_runs: int = 30,
        num_samples: int = 10_000,
        num_restarts: int = 30,
        num_neighbors: int = 50,
        refine1_iterations: int = 100,
        refine1_samples: int = 1000,
        refine2_clusters: int = 500,
        smoothing: float = 0.5,
        verbose: bool = False,
    ) -> None:
        self.num_clusters = num_clusters
        self.num_runs = num_runs
        self.num_samples = num_samples
        self.num_restarts = num_restarts
        self.num_neighbors = num_neighbors
        self.refine1_iterations = refine1_iterations
        self.refine1_samples = refine1_samples
        self.refine2_clusters = refine2_clusters
        self.smoothing = smoothing
        self.verbose = verbose
        self.mean_x = self.mean_y = self.weight = None

    @torch.inference_mode()
    @torch_threads()
    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None = None,
        paired_y: Tensor | None = None,
        randomness: Randomness | int = 0,
    ) -> "MiniVec2Vec":
        if paired_x is not None and paired_x.shape[0] > 0:
            raise ValueError("mini-vec2vec is an unpaired baseline; it takes no known pairs.")
        if not isinstance(randomness, Randomness):
            randomness = Randomness(randomness)
        samples_x, samples_y = samples_x.float(), samples_y.float()
        self.mean_x = samples_x.mean(dim=0, keepdim=True)
        self.mean_y = samples_y.mean(dim=0, keepdim=True)
        samples_x = center_and_normalize(samples_x, self.mean_x)
        samples_y = center_and_normalize(samples_y, self.mean_y)

        anchors_x, anchors_y = self.anchors(samples_x, samples_y, randomness)
        weight = self.readout(samples_x, samples_y, anchors_x, anchors_y)
        for _ in trange(self.refine1_iterations, desc="refine1", disable=not self.verbose, leave=False):
            weight = self.refine1_step(samples_x, samples_y, weight, randomness)
        self.weight = self.refine2_step(samples_x, samples_y, weight, randomness)
        return self

    def anchors(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        randomness: Randomness,
    ) -> tuple[Tensor, Tensor]:
        """Matched k-means centers of ``num_runs`` random subsets of both spaces, concatenated."""
        anchors_x, anchors_y = [], []
        for _ in trange(self.num_runs, desc="anchors", disable=not self.verbose, leave=False):
            batch_x = samples_x[randomness.subset(samples_x.shape[0], self.num_samples)]
            batch_y = samples_y[randomness.subset(samples_y.shape[0], self.num_samples)]
            centers_x = kmeans(batch_x, self.num_clusters, randomness.kmeans_seed())
            centers_y = kmeans(batch_y, self.num_clusters, randomness.kmeans_seed())
            anchors_x.append(centers_x)
            anchors_y.append(centers_y[match_clusters(centers_x, centers_y, self.num_restarts, randomness)])
        return torch.cat(anchors_x), torch.cat(anchors_y)

    def readout(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        anchors_x: Tensor,
        anchors_y: Tensor,
    ) -> Tensor:
        """Procrustes on pseudo-pairs found by nearest relative representations."""
        relative_x = samples_x @ normalize(anchors_x, dim=-1).T
        relative_y = samples_y @ normalize(anchors_y, dim=-1).T
        neighbors = cosine_neighbors(relative_x, relative_y, self.num_neighbors)
        return polar(samples_x.T @ mean_of_neighbors(neighbors, samples_y))

    def refine1_step(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        weight: Tensor,
        randomness: Randomness,
    ) -> Tensor:
        batch = samples_x[randomness.subset(samples_x.shape[0], self.refine1_samples)]
        neighbors = cosine_neighbors(batch @ weight, samples_y, self.num_neighbors)
        return self.smoothed(weight, polar(batch.T @ mean_of_neighbors(neighbors, samples_y)))

    def refine2_step(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        weight: Tensor,
        randomness: Randomness,
    ) -> Tensor:
        shuffled_x = samples_x[randomness.permutation(samples_x.shape[0])]
        shuffled_y = samples_y[randomness.permutation(samples_y.shape[0])]
        centers_x = kmeans(shuffled_x, self.refine2_clusters, randomness.kmeans_seed())
        centers_y = kmeans(shuffled_y, self.refine2_clusters, randomness.kmeans_seed(), init=centers_x @ weight)
        return self.smoothed(weight, polar(centers_x.T @ centers_y))

    def smoothed(
        self,
        weight: Tensor,
        new_weight: Tensor,
    ) -> Tensor:
        return (1 - self.smoothing) * weight + self.smoothing * new_weight

    @torch.inference_mode()
    def similarity(
        self,
        queries_x: Tensor,
        keys_y: Tensor,
    ) -> Tensor:
        mapped = center_and_normalize(queries_x.float(), self.mean_x) @ self.weight
        return mapped @ center_and_normalize(keys_y.float(), self.mean_y).T
