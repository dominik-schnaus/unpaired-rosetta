"""MLIP <-> MLIP (MP-20): one crystal structure described by two machine-learning interatomic potentials.

Row ``i`` of both files is structure ``i`` of the MP-20 training split (the first 12,000 rows of CDVAE's
``train.csv``, Materials Project entries with at most 20 atoms per cell). Each potential, MACE-MP-0 small and
MACE-MP-0 medium, describes every atom by its invariant features of all layers. A structure is the mean over its
atoms, and each side is reduced to its 256 leading principal components (``modalities/pca.py``). The two sides share
the structure but not the model, so the pair tests whether two potentials of different size describe matter alike.

    pixi run -e modalities python -m modalities.interatomic_potentials.download      # train.csv (35 MB)
    pixi run -e mlip python modalities/interatomic_potentials/descriptors.py        # MACE forward passes (GPU)
    pixi run python -m modalities.interatomic_potentials.embed                       # PCA -> embeddings/MP20-structure/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modalities.common import EmbeddingFile, RawData
from unpaired_rosetta.embeddings import storage_root

if TYPE_CHECKING:
    from pathlib import Path

DATASET = "MP20-structure"
DATA_ROOT = storage_root() / "data" / "interatomic_potentials"
STRUCTURES_CSV = DATA_ROOT / "raw" / "train.csv"
DESCRIPTORS = DATA_ROOT / "cache" / "structure"  # written by descriptors.py: <potential>.npy [structures, 256] float32

NUM_STRUCTURES = 12_000
NUM_COMPONENTS = 256


def embedding_file(
    potential: str,
) -> EmbeddingFile:
    return EmbeddingFile(
        DATASET, potential, f"{potential}-pca{NUM_COMPONENTS}-structure", (NUM_STRUCTURES, NUM_COMPONENTS), "float32"
    )


SMALL = embedding_file("mace_mp_small")
MEDIUM = embedding_file("mace_mp_medium")
PAIR = (SMALL, MEDIUM)
FILES = [SMALL, MEDIUM]

RAW_DATA = RawData("MP-20 train split of CDVAE (CIF strings)", download_bytes=34_979_124, stored_bytes=34_979_124)


def descriptor_path(
    potential: str,
) -> Path:
    return DESCRIPTORS / f"{potential}.npy"
