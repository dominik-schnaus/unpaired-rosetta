"""Single-cell biology: two assays of the same cells (PBMC, SNARE-seq, CITE-seq) and two species (human <-> mouse).

The features of the paired assays are the ones their benchmarks distribute (SCOT and SCOT+, Demetci et al.). The
human and mouse cells come from the CZ CELLxGENE Census and are embedded per species. Every file is a float32 tensor
``[cells, features]`` at ``embeddings/<dataset>/<modality>/<model>.pt``. ``prepare.py`` writes the paired assays and
``cross_species.py`` the two species.

Cell types are stored for evaluation only, as indices into the sorted cell-type names: ``labels.pt`` for the paired
assays (the same cell on both sides) and ``labels_<species>.pt`` for the two species (different cells).
"""

from modalities.common import EmbeddingFile, RawData

# dataset: ((modality, model) of space X, (modality, model) of space Y)
PAIRS = {
    "PBMC": (("rna", "pca50"), ("atac", "topics50")),  # 50 principal components <-> 50 cisTopic topics
    "SNARE-seq": (("rna", "pca10"), ("atac", "lsi19")),  # 10 principal components <-> 19 LSI components
    "CITE-seq": (("rna", "raw25"), ("adt", "raw25")),  # 25 genes <-> the 25 surface proteins they encode
    "CrossSpecies-SC": (("human", "pca50"), ("mouse", "pca50")),  # per-species PCA, different cells
}
NUM_CELLS = {"PBMC": 2407, "SNARE-seq": 1047, "CITE-seq": 1000, "CrossSpecies-SC": 34500}
DIMENSIONS = {"pca50": 50, "topics50": 50, "pca10": 10, "lsi19": 19, "raw25": 25}
PAIRED_DATASETS = ["PBMC", "SNARE-seq", "CITE-seq"]  # the same cells in both assays


def embedding_files(
    dataset: str,
) -> list[EmbeddingFile]:
    return [
        EmbeddingFile(dataset, modality, model, (NUM_CELLS[dataset], DIMENSIONS[model]), "float32")
        for modality, model in PAIRS[dataset]
    ]


def label_files(
    dataset: str,
) -> list[str]:
    """Cell-type files, relative to ``embeddings/<dataset>/``."""
    if dataset in PAIRED_DATASETS:
        return ["labels.pt"]
    return [f"labels_{modality}.pt" for modality, _ in PAIRS[dataset]]


RAW_DATA = {
    "PBMC": RawData("SCOT+ PBMC tables (10x Multiome, 2407 cells)", download_bytes=17_400_000, stored_bytes=17_400_000),
    "SNARE-seq": RawData("SCOT SNARE-seq features (1047 cells)", download_bytes=250_000, stored_bytes=250_000),
    "CITE-seq": RawData("SCOT+ CITE-seq features (1000 cells)", download_bytes=1_250_000, stored_bytes=1_250_000),
    "CrossSpecies-SC": RawData(
        "CELLxGENE Census raw counts (2 x 34,500 cells)", download_bytes=1_300_000_000, stored_bytes=1_300_000_000
    ),
}
