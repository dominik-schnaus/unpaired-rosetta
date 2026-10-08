"""Orthogonal map fitted on the known pairs only (Maiorca et al., 2023).

Both sides are centered and normalized like in our aligner, and the map is ``W = polar(X_hat^T Y_hat)`` on the
``p`` pairs. Without known pairs the similarity is constant (chance level).
"""

import torch
from torch import Tensor

from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.linalg import center_and_normalize, polar
from unpaired_rosetta.randomness import Randomness


class OrthogonalMap:
    def __init__(
        self,
    ) -> None:
        self.mean_x = self.mean_y = self.weight = None

    @torch.inference_mode()
    @torch_threads()
    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None = None,
        paired_y: Tensor | None = None,
        randomness: Randomness | None = None,
    ) -> "OrthogonalMap":
        self.mean_x = samples_x.mean(dim=0, keepdim=True)
        self.mean_y = samples_y.mean(dim=0, keepdim=True)
        if paired_x is not None and paired_x.shape[0] > 0:
            paired_x = center_and_normalize(paired_x, self.mean_x)
            paired_y = center_and_normalize(paired_y, self.mean_y)
            self.weight = polar(paired_x.T @ paired_y)
        return self

    @torch.inference_mode()
    def similarity(
        self,
        queries_x: Tensor,
        keys_y: Tensor,
    ) -> Tensor:
        if self.weight is None:
            return torch.zeros(queries_x.shape[0], keys_y.shape[0])
        mapped = center_and_normalize(queries_x, self.mean_x) @ self.weight
        return mapped @ center_and_normalize(keys_y, self.mean_y).T
