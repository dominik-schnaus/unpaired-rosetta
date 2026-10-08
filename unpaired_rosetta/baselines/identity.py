"""The identity map, which compares the two spaces in their own coordinates.

It is the control for pairs whose two sides come from the same encoder (the CycleGAN pairs), where there is no
rotation to find. With ``centered=True`` both sides are first centered and normalized like in our aligner. Both
sides must have the same width.
"""

import torch
from torch import Tensor

from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.linalg import center_and_normalize
from unpaired_rosetta.randomness import Randomness


class Identity:
    def __init__(
        self,
        centered: bool = False,
    ) -> None:
        self.centered = centered
        self.mean_x = self.mean_y = None

    @torch.inference_mode()
    @torch_threads()
    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None = None,
        paired_y: Tensor | None = None,
        randomness: Randomness | None = None,
    ) -> "Identity":
        if samples_x.shape[1] != samples_y.shape[1]:
            widths = f"{samples_x.shape[1]} and {samples_y.shape[1]}"
            raise ValueError(f"The identity map needs two spaces of equal width, got {widths}.")
        if self.centered:
            self.mean_x = samples_x.mean(dim=0, keepdim=True)
            self.mean_y = samples_y.mean(dim=0, keepdim=True)
        return self

    @torch.inference_mode()
    def similarity(
        self,
        queries_x: Tensor,
        keys_y: Tensor,
    ) -> Tensor:
        if self.centered:
            queries_x = center_and_normalize(queries_x, self.mean_x)
            keys_y = center_and_normalize(keys_y, self.mean_y)
        return queries_x @ keys_y.T
