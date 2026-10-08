"""Wasserstein Procrustes with one of its three stages replaced, for the component ablation (Tables 1, 3 and 4).

The stages that are not replaced are ours and use the same random draws in the same order. The ablation is
unpaired.
"""

import torch
from torch import Tensor

from unpaired_rosetta.ablation.initializations import (
    ClusterMatching,
    GromovWasserstein,
    PCAHeuristic,
    SortedHeuristic,
    average_cross_covariance,
)
from unpaired_rosetta.ablation.readouts import direct_readout, relative_representation_readout
from unpaired_rosetta.ablation.refinements import mini_vec2vec_refinement
from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.geometric_initialization import GeometricInitialization
from unpaired_rosetta.linalg import center_and_normalize
from unpaired_rosetta.randomness import Randomness
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

INITIALIZATIONS = ["PCA", "sorted", "GW (random)", "GW (uniform)", "2-opt", "ours"]
READOUTS = ["direct", "relative representations", "ours"]
REFINEMENTS = ["none", "mini-vec2vec", "ours"]


class AblatedWassersteinProcrustes(WassersteinProcrustes):
    """Wasserstein Procrustes where ``initialization``, ``readout`` and ``refinement`` name the variant of each stage.

    ``"ours"`` keeps a stage unchanged. ``"2-opt"`` is mini-vec2vec's cluster matching, which is our clustering
    with 2-opt instead of MPOpt. The relative-representation read-out needs the matched centers of a cluster matching.
    """

    def __init__(
        self,
        initialization: str = "ours",
        readout: str = "ours",
        refinement: str = "ours",
        **kwargs: int | bool,
    ) -> None:
        super().__init__(**kwargs)
        if initialization not in INITIALIZATIONS or readout not in READOUTS or refinement not in REFINEMENTS:
            raise ValueError(f"Unknown variant {(initialization, readout, refinement)}")
        if readout == "relative representations" and initialization not in ("2-opt", "ours"):
            raise ValueError("The relative-representation read-out needs matched cluster centers.")
        self.initialization = initialization
        self.readout_variant = readout
        self.refinement = refinement

    @torch.inference_mode()
    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None = None,
        paired_y: Tensor | None = None,
        randomness: Randomness | int = 0,
    ) -> "AblatedWassersteinProcrustes":
        if paired_x is not None and paired_x.shape[0] > 0:
            raise ValueError("The component ablation is unpaired.")
        if not isinstance(randomness, Randomness):
            randomness = Randomness(randomness)
        with torch_threads():
            samples_x, samples_y = self.preprocess(samples_x, samples_y)
            correspondence, anchors = self.initialize(samples_x, samples_y, randomness)
            weight = self.read_out(samples_x, samples_y, correspondence, anchors, randomness)
            self.weight = self.refine(samples_x, samples_y, weight, randomness)
        return self

    def preprocess(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Center and normalize both sides, as ``WassersteinProcrustes.fit`` does."""
        samples_x, samples_y = samples_x.float(), samples_y.float()
        self.mean_x = samples_x.mean(dim=0, keepdim=True)
        self.mean_y = samples_y.mean(dim=0, keepdim=True)
        return center_and_normalize(samples_x, self.mean_x), center_and_normalize(samples_y, self.mean_y)

    def initialize(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        randomness: Randomness,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """The coarse correspondence ``M`` and, for cluster matching, the matched centers of all restarts."""
        heuristics = {
            "PCA": PCAHeuristic(),
            "sorted": SortedHeuristic(),
            "GW (random)": GromovWasserstein("random"),
            "GW (uniform)": GromovWasserstein("uniform"),
        }
        if self.initialization in heuristics:
            return heuristics[self.initialization](samples_x, samples_y, randomness), None
        if self.initialization == "ours" and self.readout_variant != "relative representations":
            initialization = GeometricInitialization(
                self.num_clusters, self.num_restarts, self.batch_size, self.num_workers
            )
            return initialization(samples_x, samples_y, None, None, randomness), None
        solver = "MPOpt" if self.initialization == "ours" else "2-opt"
        matching = ClusterMatching(self.num_clusters, solver, self.num_restarts, self.batch_size, self.num_workers)
        matched_centers = matching(samples_x, samples_y, randomness)
        anchors = (
            torch.cat([centers for centers, _ in matched_centers]),
            torch.cat([centers for _, centers in matched_centers]),
        )
        return average_cross_covariance(matched_centers), anchors

    def read_out(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        correspondence: Tensor,
        anchors: tuple[Tensor, Tensor] | None,
        randomness: Randomness,
    ) -> Tensor:
        if self.readout_variant == "direct":
            return direct_readout(correspondence)
        if self.readout_variant == "relative representations":
            return relative_representation_readout(samples_x, samples_y, *anchors)
        return self.readout(samples_x, samples_y, correspondence, None, None, randomness)

    def refine(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        weight: Tensor,
        randomness: Randomness,
    ) -> Tensor:
        if self.refinement == "mini-vec2vec":
            return mini_vec2vec_refinement(samples_x, samples_y, weight, randomness)
        if self.refinement == "ours":
            for _ in range(self.num_iterations):
                weight = self.refinement_step(samples_x, samples_y, weight, None, None, randomness)
        return weight
