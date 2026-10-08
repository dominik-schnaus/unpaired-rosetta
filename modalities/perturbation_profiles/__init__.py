"""Cell Painting <-> L1000 (Rosetta, LINCS Pilot 1): the morphology and the transcription of cells
under one perturbation.

Row ``i`` of both files is the same (compound, dose) unit: the consensus (median) Cell Painting profile of its wells
against the consensus L1000 profile of its signatures. Imaging and sequencing are separate assays on separate plates,
so the two sides share the perturbation and nothing else. Both assays use A549 cells. Each side is reduced to its
leading principal components (``modalities/pca.py``): 512 of the 1,783 CellProfiler features, 256 of the 978 L1000
landmark genes.

    pixi run -e modalities python -m modalities.perturbation_profiles.download   # replicate profiles (475 MB)
    pixi run -e modalities python -m modalities.perturbation_profiles.profiles   # consensus profiles of 2,732 units
    pixi run python -m modalities.perturbation_profiles.embed                    # PCA: embeddings/Rosetta-LINCS-Pilot1/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modalities.common import EmbeddingFile, RawData
from unpaired_rosetta.embeddings import storage_root

if TYPE_CHECKING:
    from pathlib import Path

DATASET = "Rosetta-LINCS-Pilot1"
DATA_ROOT = storage_root() / "data" / "perturbation_profiles"
REPLICATES = {  # replicate-level profiles of the Rosetta resource (cpg0003-rosetta), per assay
    "morphology": DATA_ROOT / "raw" / "LINCS-Pilot1" / "CellPainting" / "replicate_level_cp_augmented.parquet",
    "expression": DATA_ROOT / "raw" / "LINCS-Pilot1" / "L1000" / "replicate_level_l1k.parquet",
}
PROFILES = DATA_ROOT / "cache" / "LINCS-Pilot1"  # written by profiles.py: <assay>.npy [units, features], index.json

NUM_UNITS = 2732
MORPHOLOGY = EmbeddingFile(DATASET, "morphology", "morphology-pca512", (NUM_UNITS, 512), "float32")
EXPRESSION = EmbeddingFile(DATASET, "expression", "expression-pca256", (NUM_UNITS, 256), "float32")
PAIR = (MORPHOLOGY, EXPRESSION)
FILES = [MORPHOLOGY, EXPRESSION]

RAW_DATA = RawData(
    "Rosetta LINCS Pilot 1 replicate profiles (Cell Painting and L1000)",
    download_bytes=474_854_042,
    stored_bytes=474_854_042,
)


def profile_path(
    assay: str,
) -> Path:
    return PROFILES / f"{assay}.npy"
