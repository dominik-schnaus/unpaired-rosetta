"""ASIF (Norelli et al., 2023), which compares two spaces through relative representations to the known pairs.

Each sample is described by its cosine similarities to the ``p`` anchors of its space. As in the official code, the
``k`` largest are kept, raised to the power ``exponent`` and normalized. The score is the inner product of these
descriptions. Unlike the official code, negative similarities are set to zero first, since an even power would turn
a dissimilar anchor into a similar one. The difference matters when ``k`` exceeds ``p`` and all anchors are kept.
Without known pairs the similarity is constant (chance level).
"""

import torch
from torch import Tensor
from torch.nn.functional import normalize

from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.linalg import center_and_normalize
from unpaired_rosetta.randomness import Randomness


def cosine_similarity(
    queries: Tensor,
    keys: Tensor,
) -> Tensor:
    return normalize(queries, dim=-1) @ normalize(keys, dim=-1).T


class ASIF:
    def __init__(
        self,
        num_neighbors: int = 800,
        exponent: int = 8,
    ) -> None:
        self.num_neighbors = num_neighbors  # k ("non_zeros" in the official code)
        self.exponent = exponent  # "val_exp" in the official code
        self.mean_x = self.mean_y = self.anchors_x = self.anchors_y = None

    @torch.inference_mode()
    @torch_threads()
    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None = None,
        paired_y: Tensor | None = None,
        randomness: Randomness | None = None,
    ) -> "ASIF":
        self.mean_x = samples_x.mean(dim=0, keepdim=True)
        self.mean_y = samples_y.mean(dim=0, keepdim=True)
        if paired_x is not None and paired_x.shape[0] > 0:
            self.anchors_x = center_and_normalize(paired_x, self.mean_x)
            self.anchors_y = center_and_normalize(paired_y, self.mean_y)
        return self

    def relative_representation(
        self,
        samples: Tensor,
        anchors: Tensor,
    ) -> Tensor:
        """Top-``k`` cosine similarities to the anchors, clipped at zero, raised to ``exponent`` and normalized."""
        similarity = cosine_similarity(samples, anchors)
        indices = similarity.topk(k=min(self.num_neighbors, anchors.shape[0]), dim=1).indices
        sparse = torch.zeros_like(similarity).scatter_(1, indices, similarity.gather(1, indices))
        return normalize(sparse.clamp(min=0).pow(self.exponent), dim=1)

    @torch.inference_mode()
    def similarity(
        self,
        queries_x: Tensor,
        keys_y: Tensor,
    ) -> Tensor:
        if self.anchors_x is None:
            return torch.zeros(queries_x.shape[0], keys_y.shape[0])
        relative_x = self.relative_representation(center_and_normalize(queries_x, self.mean_x), self.anchors_x)
        relative_y = self.relative_representation(center_and_normalize(keys_y, self.mean_y), self.anchors_y)
        return relative_x @ relative_y.T
