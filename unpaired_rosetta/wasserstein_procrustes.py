"""Wasserstein Procrustes with geometric initialization (Algorithm 1 of the paper).

Solves ``max_{W in St(d_X, d_Y), T in Pi(a, b)} Tr(X W Y^T T^T)`` for a semi-orthogonal map ``W`` and a correspondence
``T``. It alternates linear assignments for ``T`` and orthogonal Procrustes for ``W`` on random batches, starting from
the geometric initialization. Known pairs, if given, are added to every Procrustes step.
"""

import torch
from torch import Tensor
from tqdm.auto import trange

from unpaired_rosetta.assignment import batched_hungarian_matching
from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.geometric_initialization import GeometricInitialization
from unpaired_rosetta.linalg import center_and_normalize, polar
from unpaired_rosetta.randomness import Randomness


def procrustes(
    matched_x: Tensor,
    matched_y: Tensor,
    paired_x: Tensor | None,
    paired_y: Tensor | None,
) -> Tensor:
    """``polar(X^T T Y + (k/p) X_hat^T Y_hat)`` for ``k`` matched rows and ``p`` known pairs.

    Lines 5 and 9 of Algorithm 1. The weight ``k/p`` gives the known pairs the same total weight as the pseudo-pairs.
    """
    cross_covariance = matched_x.T @ matched_y
    if paired_x is not None:
        pair_weight = matched_x.shape[0] / paired_x.shape[0]
        cross_covariance = cross_covariance + (pair_weight * paired_x.T) @ paired_y
    return polar(cross_covariance)


class WassersteinProcrustes:
    """The aligner of the paper: a semi-orthogonal map ``W`` from X to Y fitted without paired data.

    Args:
        num_clusters: Number of k-means centers ``C`` per restart of the initialization.
        num_restarts: Number of restarts ``S`` of the initialization.
        batch_size: Batch size ``b`` of the initialization and of every refinement step.
        num_iterations: Number of refinement steps ``R``.
        num_workers: Processes for the initialization restarts. Does not change the result.
        num_threads: Threads for the assignments of the readout. Does not change the result.
        verbose: Show progress bars.

    Attributes:
        mean_x, mean_y: Training means ``[1, d]`` of both spaces, set by ``fit``.
        weight: The fitted map ``[d_X, d_Y]``, set by ``fit``.
    """

    def __init__(
        self,
        num_clusters: int = 30,  # C
        num_restarts: int = 30,  # S
        batch_size: int = 10_000,  # b
        num_iterations: int = 100,  # R
        num_workers: int = 1,
        num_threads: int = 1,
        verbose: bool = False,
    ) -> None:
        self.num_clusters = num_clusters
        self.num_restarts = num_restarts
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.num_workers = num_workers
        self.num_threads = num_threads
        self.verbose = verbose
        self.mean_x = self.mean_y = self.weight = None

    @torch.inference_mode()
    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None = None,
        paired_y: Tensor | None = None,
        randomness: Randomness | int = 0,
    ) -> "WassersteinProcrustes":
        """Fit the map on two unpaired sets and optional known pairs.

        Args:
            samples_x: Unpaired samples of X ``[n, d_X]``.
            samples_y: Unpaired samples of Y ``[m, d_Y]``.
            paired_x: Known pairs in X ``[p, d_X]``, row-aligned with ``paired_y``, or None.
            paired_y: Known pairs in Y ``[p, d_Y]``, or None.
            randomness: A seed or the ``Randomness`` of the run.

        Returns:
            The fitted aligner itself.
        """
        if not isinstance(randomness, Randomness):
            randomness = Randomness(randomness)
        with torch_threads():
            return self._fit(samples_x, samples_y, paired_x, paired_y, randomness)

    def _fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None,
        paired_y: Tensor | None,
        randomness: Randomness,
    ) -> "WassersteinProcrustes":
        samples_x, samples_y = samples_x.float(), samples_y.float()
        self.mean_x = samples_x.mean(dim=0, keepdim=True)
        self.mean_y = samples_y.mean(dim=0, keepdim=True)
        samples_x = center_and_normalize(samples_x, self.mean_x)
        samples_y = center_and_normalize(samples_y, self.mean_y)
        if paired_x is not None and paired_x.shape[0] > 0:
            paired_x = center_and_normalize(paired_x.float(), self.mean_x)
            paired_y = center_and_normalize(paired_y.float(), self.mean_y)
        else:
            paired_x = paired_y = None

        initialization = GeometricInitialization(
            self.num_clusters, self.num_restarts, self.batch_size, self.num_workers
        )
        correspondence = initialization(samples_x, samples_y, paired_x, paired_y, randomness)
        weight = self.readout(samples_x, samples_y, correspondence, paired_x, paired_y, randomness)
        for _ in trange(self.num_iterations, desc="refinement", disable=not self.verbose, leave=False):
            weight = self.refinement_step(samples_x, samples_y, weight, paired_x, paired_y, randomness)
        self.weight = weight
        return self

    def readout(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        correspondence: Tensor,
        paired_x: Tensor | None,
        paired_y: Tensor | None,
        randomness: Randomness,
    ) -> Tensor:
        """First map from the averaged plan (line 9 of Algorithm 2 and line 5 of Algorithm 1).

        Matches on the scores ``X M Y^T`` in random blocks of ``b`` rows, then solves Procrustes on the pseudo-pairs.
        """
        queries = samples_x @ correspondence
        order_x = randomness.permutation(samples_x.shape[0])
        order_y = randomness.permutation(samples_y.shape[0])
        rows, columns = batched_hungarian_matching(
            queries[order_x], samples_y[order_y], self.batch_size, self.num_threads
        )
        return procrustes(samples_x[order_x[rows]], samples_y[order_y[columns]], paired_x, paired_y)

    def refinement_step(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        weight: Tensor,
        paired_x: Tensor | None,
        paired_y: Tensor | None,
        randomness: Randomness,
    ) -> Tensor:
        """Match a random batch of ``X W`` to a random batch of ``Y``, then solve Procrustes.

        Lines 7-9 of Algorithm 1.
        """
        batch_x = samples_x[randomness.subset(samples_x.shape[0], self.batch_size)]
        batch_y = samples_y[randomness.subset(samples_y.shape[0], self.batch_size)]
        block_size = max(batch_x.shape[0], batch_y.shape[0])
        rows, columns = batched_hungarian_matching(batch_x @ weight, batch_y, block_size)
        return procrustes(batch_x[rows], batch_y[columns], paired_x, paired_y)

    @torch.inference_mode()
    def transform(
        self,
        samples_x: Tensor,
    ) -> Tensor:
        """Map samples of X into the centered and normalized coordinates of Y."""
        return center_and_normalize(samples_x.float(), self.mean_x) @ self.weight

    @torch.inference_mode()
    def similarity(
        self,
        queries_x: Tensor,
        keys_y: Tensor,
    ) -> Tensor:
        """Inner products ``x W y^T`` of the preprocessed samples, which equal their cosine similarities."""
        return self.transform(queries_x) @ center_and_normalize(keys_y.float(), self.mean_y).T
