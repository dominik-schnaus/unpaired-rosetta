"""Galaxy image <-> spectrum (AstroCLIP DESI): a Legacy Survey image and a DESI optical spectrum of the same galaxy.

Uses the test split (29,697 galaxies) of the AstroCLIP cross-match of DESI Early Data Release spectra with Legacy
Imaging Survey cutouts (Parker et al., 2024). Row ``i`` of both files is galaxy ``i`` of that split. AstroCLIP starts
from two single-modality encoders before it aligns them with pairs, and both are used here:

* images: AstroDINO (a DINOv2 ViT-L/12 trained only on Legacy Survey images), CLS token.
* spectra: SpecFormer (a transformer trained by masked modelling only on spectra), mean over positions.

Both modalities encode the redshift well. ``redshift_deciles`` gives equal-count redshift groups that show what an
alignment recovers (Fig. 19).

    pixi run python -m modalities.galaxy_spectra.download   # test split, stretched images, checkpoints
    pixi run python -m modalities.galaxy_spectra.embed      # embeddings/AstroCLIP-DESI/{image,spectrum}/
"""

import json

import numpy as np
import torch
from torch import Tensor

from modalities.common import EmbeddingFile, RawData
from unpaired_rosetta.embeddings import storage_root

DATASET = "AstroCLIP-DESI"
DATA_ROOT = storage_root() / "data" / "galaxy_spectra"
CACHE = DATA_ROOT / "cache"  # images.npy [N, 152, 152, 3] float16 RGB, spectra.npy [N, 7781] float32, index.json
CHECKPOINTS = DATA_ROOT / "pretrained"  # astrodino.ckpt, specformer.ckpt

NUM_GALAXIES = 29_697
NUM_DECILES = 10
IMAGE = EmbeddingFile(DATASET, "image", "astrodino", (NUM_GALAXIES, 1024), "float32")
SPECTRUM = EmbeddingFile(DATASET, "spectrum", "specformer", (NUM_GALAXIES, 768), "float32")
PAIR = (IMAGE, SPECTRUM)
FILES = [IMAGE, SPECTRUM]

# The 26 Parquet shards of the test split are streamed and not kept. The checkpoints are 1.48 GB of the stored bytes.
RAW_DATA = RawData(
    "AstroCLIP DESI test split (streamed) and the AstroDINO and SpecFormer checkpoints",
    download_bytes=10_735_440_470,
    stored_bytes=6_522_329_353,
)


def redshifts() -> np.ndarray:
    """Redshift of each galaxy, in row order."""
    return np.array([entry["redshift"] for entry in json.loads((CACHE / "index.json").read_text())])


def redshift_deciles() -> tuple[Tensor, list[str]]:
    """Equal-count redshift groups. Returns the group index of each galaxy and the names ``z0`` to ``z9``."""
    values = redshifts()
    edges = np.quantile(values, np.linspace(0, 1, NUM_DECILES + 1)[1:-1])
    return torch.as_tensor(np.searchsorted(edges, values)), [f"z{group}" for group in range(NUM_DECILES)]
