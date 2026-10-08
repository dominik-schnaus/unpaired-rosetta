"""Two other ways to turn the coarse correspondence into a first map (Table 3).

* Direct: one Procrustes step on the averaged plan, ``W = polar(M)`` (Alvarez-Melis & Jaakkola, 2018).
* Top-k relative representations (mini-vec2vec, Dar, 2025): describe each sample by its inner products with the
  matched cluster centers of its space, pair it with the ``k = 50`` most similar samples of the other space, and
  solve Procrustes on the means of those neighbors. As in the runs behind Table 3, the centers are not normalized.
  The official notebook and our mini-vec2vec baseline normalize them, which weighs the anchors differently.

Ours is ``WassersteinProcrustes.readout``.
"""

from torch import Tensor

from unpaired_rosetta.baselines.mini_vec2vec import cosine_neighbors, mean_of_neighbors
from unpaired_rosetta.linalg import polar

NUM_NEIGHBORS = 50


def direct_readout(
    correspondence: Tensor,
) -> Tensor:
    return polar(correspondence)


def relative_representation_readout(
    samples_x: Tensor,
    samples_y: Tensor,
    anchors_x: Tensor,
    anchors_y: Tensor,
) -> Tensor:
    """``polar(X^T T Y)``, where ``T`` spreads each sample of X evenly over its top-``k`` neighbors in Y.

    Neighbors are found by cosine similarity of ``X A^T`` and ``Y B^T``. ``A`` and ``B`` are row-aligned anchors,
    here the matched centers of all restarts.
    """
    neighbors = cosine_neighbors(samples_x @ anchors_x.T, samples_y @ anchors_y.T, NUM_NEIGHBORS)
    return polar(samples_x.T @ mean_of_neighbors(neighbors, samples_y))
