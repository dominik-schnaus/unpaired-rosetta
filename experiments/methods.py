"""The aligners compared in the experiments, by the name used in the result files and figures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from unpaired_rosetta.baselines.asif import ASIF
from unpaired_rosetta.baselines.identity import Identity
from unpaired_rosetta.baselines.linear import LinearMap
from unpaired_rosetta.baselines.local_cka import LocalCKA
from unpaired_rosetta.baselines.mini_vec2vec import MiniVec2Vec
from unpaired_rosetta.baselines.orthogonal import OrthogonalMap
from unpaired_rosetta.baselines.scot_plus import SCOTPlus
from unpaired_rosetta.baselines.sotalign import SOTAlign
from unpaired_rosetta.baselines.structure import STRUCTURE
from unpaired_rosetta.baselines.sue import SUE
from unpaired_rosetta.baselines.vec2vec import Vec2Vec
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

if TYPE_CHECKING:
    from unpaired_rosetta.evaluation import Aligner


def gpu_if_available() -> str:
    """Use a GPU if there is one, as the paper's runs of the trained baselines did."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# Baselines with the hyperparameters of the paper's runs, which are their constructors' defaults.
BASELINES = {
    # unpaired alignment (Secs. 4.1 and 4.2)
    "vec2vec": lambda: Vec2Vec(device=gpu_if_available()),
    "mini-vec2vec": MiniVec2Vec,
    # few-pair alignment (Sec. 4.4)
    "linear": LinearMap,
    "orthogonal": OrthogonalMap,
    "ASIF": ASIF,
    "LocalCKA": LocalCKA,
    "SUE": lambda: SUE(device=gpu_if_available()),
    "STRUCTURE": lambda: STRUCTURE(device=gpu_if_available()),
    "SOTAlign": lambda: SOTAlign(device=gpu_if_available()),
    # single-cell alignment (PBMC, Sec. 4.2)
    "SCOT+": lambda: SCOTPlus(device=gpu_if_available()),
    # CycleGAN pairs, where both sides come from one encoder
    "identity": lambda: Identity(centered=False),
    "identity (centred+norm)": lambda: Identity(centered=True),
}


def build_aligner(
    method: str,
    num_workers: int = 1,
    num_threads: int = 1,
) -> Aligner:
    """A new, unfitted aligner. Workers and threads change only the speed, never the result."""
    if method == "ours":
        return WassersteinProcrustes(num_workers=num_workers, num_threads=num_threads)
    if method in BASELINES:
        return BASELINES[method]()
    raise ValueError(f"Unknown method {method!r}")
