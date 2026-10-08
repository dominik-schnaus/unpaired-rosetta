"""Histology <-> expression (HEST-Benchmark, CCRCC): the H&E image and the transcriptome of the same Visium spot.

Every spot of the 24 clear cell renal cell carcinoma slides of HEST-Benchmark is measured twice: a 224 x 224 pixel
tile of the H&E stain at 0.5 um per pixel centred on the spot (112 um), and the RNA counts of the spot. The two
are joined on the spot barcode, so row ``i`` of both files is the same spot (74,220 spots). The tile is embedded
with H-Optimus-0. The expression is log1p counts per 10,000 over the 2,000 most variable genes of the cohort,
reduced to 64 principal components (``modalities/pca.py``).

    pixi run -e modalities python -m modalities.histology_expression.download  # 13.9 GB from the Hugging Face Hub
    pixi run -e modalities python -m modalities.histology_expression.prepare   # tiles.npy and expression.npy
    pixi run -e modalities python -m modalities.histology_expression.embed     # embeddings/HEST-CCRCC/
"""

from modalities.common import EmbeddingFile, RawData
from unpaired_rosetta.embeddings import storage_root

DATASET = "HEST-CCRCC"
COHORT = "CCRCC"
DATA_ROOT = storage_root() / "data" / "histology_expression"
RAW = DATA_ROOT / "raw" / COHORT  # patches/<slide>.h5 and adata/<slide>.h5ad of MahmoodLab/hest-bench
CACHE = DATA_ROOT / "cache" / COHORT  # tiles.npy, expression.npy, index.json, genes.json

NUM_SPOTS = 74_220
TILE = EmbeddingFile(DATASET, "tile", "h_optimus_0", (NUM_SPOTS, 1536), "float32")
EXPRESSION = EmbeddingFile(DATASET, "expression", "expression-pca64-ccrcc", (NUM_SPOTS, 64), "float32")
PAIR = (TILE, EXPRESSION)
FILES = [TILE, EXPRESSION]

RAW_DATA = RawData(
    "HEST-Benchmark CCRCC: 24 slides of spot tiles (11.5 GB) and counts (2.4 GB)",
    download_bytes=13_868_067_968,
    stored_bytes=25_640_549_854,
)  # raw + cache
