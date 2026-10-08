"""mini-vec2vec's refinement (Dar, 2025), compared with ours in Table 4.

``refine 1`` repeats 100 times: map a random batch of X, find the ``k = 50`` nearest samples in Y for each, and
solve Procrustes against their means. ``refine 2`` runs k-means with 500 centers on X, then on Y starting from the
mapped centers, and solves Procrustes on the paired centers. Each step updates
``W <- polar(W / 2 + polar(X^T T Y) / 2)``.

As in the runs behind Table 4, the batch holds 1024 samples and every average is projected back with ``polar``.
The official notebook uses 1000 samples and the plain average ``W / 2 + W_new / 2``. The notebook version is
``unpaired_rosetta.baselines.mini_vec2vec``. Ours is ``WassersteinProcrustes.refinement_step``.
"""

from torch import Tensor

from unpaired_rosetta.baselines.mini_vec2vec import cosine_neighbors, kmeans, mean_of_neighbors
from unpaired_rosetta.linalg import polar
from unpaired_rosetta.randomness import Randomness

NUM_ITERATIONS = 100
BATCH_SIZE = 1024
NUM_NEIGHBORS = 50
NUM_CENTERS = 500
STEP_SIZE = 0.5


def mini_vec2vec_refinement(
    samples_x: Tensor,
    samples_y: Tensor,
    weight: Tensor,
    randomness: Randomness,
) -> Tensor:
    for _ in range(NUM_ITERATIONS):
        weight = moving_average(weight, neighbor_step(samples_x, samples_y, weight, randomness))
    return moving_average(weight, cluster_step(samples_x, samples_y, weight, randomness))


def moving_average(
    weight: Tensor,
    cross_covariance: Tensor,
) -> Tensor:
    return polar((1 - STEP_SIZE) * weight + STEP_SIZE * polar(cross_covariance))


def neighbor_step(
    samples_x: Tensor,
    samples_y: Tensor,
    weight: Tensor,
    randomness: Randomness,
) -> Tensor:
    """``X^T T Y`` for a random batch of X, where ``T`` spreads each mapped sample over its top-k neighbors in Y."""
    batch = samples_x[randomness.subset(samples_x.shape[0], BATCH_SIZE)]
    neighbors = cosine_neighbors(batch @ weight, samples_y, NUM_NEIGHBORS)
    return batch.T @ mean_of_neighbors(neighbors, samples_y)


def cluster_step(
    samples_x: Tensor,
    samples_y: Tensor,
    weight: Tensor,
    randomness: Randomness,
) -> Tensor:
    """``A^T B`` for the k-means centers ``A`` of X and the centers ``B`` of Y started from ``A W``."""
    shuffled_x = samples_x[randomness.permutation(samples_x.shape[0])]
    shuffled_y = samples_y[randomness.permutation(samples_y.shape[0])]
    centers_x = kmeans(shuffled_x, NUM_CENTERS, randomness.kmeans_seed())
    centers_y = kmeans(shuffled_y, NUM_CENTERS, randomness.kmeans_seed(), init=centers_x @ weight)
    return centers_x.T @ centers_y
