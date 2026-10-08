"""Write the features of the paired single-cell benchmarks into the embedding layout.

* PBMC (SCOT+, Baker et al.): 2407 cells of the 10x Multiome PBMC data, gene expression as 50 principal components
  and chromatin accessibility as 50 cisTopic topics, with the cell types of ``adata.h5ad``.
* SNARE-seq (SCOT, Demetci et al.): 1047 cells, expression as 10 principal components and accessibility as 19 LSI
  components, with four cell lines as the cell types.
* CITE-seq (SCOT+): 1000 cells, 25 genes and the 25 surface proteins they encode (no cell types).

The features are stored as distributed, in float32, and nothing is fitted here. Needs ``h5py`` for the PBMC cell
types::

    pixi run -e modalities python -m modalities.single_cell.prepare
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from modalities.common import download
from unpaired_rosetta.embeddings import embedding_path, storage_root

if TYPE_CHECKING:
    from pathlib import Path

SCOT_URL = "https://raw.githubusercontent.com/rsinghlab/SCOT/14649be6e14017dcfe7ba619091b33d1df55f6a9/data"
SCOT_PLUS_URL = (
    "https://raw.githubusercontent.com/scotplus/book_source/87aedd0fa08c00b42539e16a994e1e3a48b2a62e/book/data"
)
CHECKSUMS = {  # SHA-256 of the files at the pinned commits
    "SNARE/SNAREseq_rna_feat.npy": "a8a2c2bbfc036dea5b350d01dd2180b93bb3b1c0008bed49493080348da96a9b",
    "SNARE/SNAREseq_atac_feat.npy": "80d685b6517c100a7fd612846ad38209b64ed9caf095d1630af5a1d7797bd79c",
    "SNARE/SNAREseq_rna_types.txt": "3697fe94aee8b0ed2fad49c10c3226b757cedd8046a5cc0b10d0a845ad540a1d",
    "PBMC/rna_50pca_50topics.pkl": "0794df6ed5f4ccedb3dd38929c7c8570f13aa3c11131c39c859b56ba8acffe5b",
    "PBMC/atac_50pca_50topics.pkl": "a959004c801b85fba73091a1f4003b377282859ff516b2b1d3842f4cad90602a",
    "PBMC/adata.h5ad": "3e87296cb2019a1702f06c8d69da59682eecafa77791f47d9751739f343bad9a",
    "CITEseq/citeseq_rna_normalizedFC_1000cells.csv": "e1ab54359f9cbdd47ac74bcf3c3f61d22da3eedcd932501ee30bab73e352855e",
    "CITEseq/citeseq_adt_normalized_1000cells.csv": "4c827685c66f52c483c0f4b65fb72f7dee18e90a36d486827a5bd325e794a0d3",
}


def raw_file(
    name: str,
) -> Path:
    """A raw file of SCOT (``SNARE/``) or of the SCOT+ book (``PBMC/``, ``CITEseq/``), downloaded on first use."""
    base = SCOT_URL if name.startswith("SNARE/") else SCOT_PLUS_URL
    return download(f"{base}/{name}", storage_root() / "raw" / "single_cell" / name, CHECKSUMS[name])


def label_indices(
    cell_types: list[str],
) -> torch.Tensor:
    """The index of every cell's type among the sorted type names."""
    names = sorted(set(cell_types))
    return torch.tensor([names.index(cell_type) for cell_type in cell_types])


def save(
    tensor: torch.Tensor,
    dataset: str,
    modality: str,
    model: str,
) -> None:
    path = embedding_path(dataset, modality, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    print(f"wrote {path} {tuple(tensor.shape)}")


def save_labels(
    labels: torch.Tensor,
    dataset: str,
) -> None:
    torch.save(labels, storage_root() / "embeddings" / dataset / "labels.pt")


def pbmc_cell_types() -> pd.Series:
    """The ``celltype`` annotation of ``adata.h5ad`` by cell barcode, read with h5py from the AnnData store."""
    import h5py

    with h5py.File(raw_file("PBMC/adata.h5ad"), "r") as file:
        cells = file["obs"]
        barcodes = cells[cells.attrs["_index"]][:].astype(str)
        names = cells["celltype"]["categories"][:].astype(str)
        codes = cells["celltype"]["codes"][:]
    return pd.Series(names[codes], index=barcodes)


def prepare_pbmc() -> None:
    with open(raw_file("PBMC/rna_50pca_50topics.pkl"), "rb") as file:
        rna = pickle.load(file)  # DataFrame [cells, 50], indexed by barcode
    with open(raw_file("PBMC/atac_50pca_50topics.pkl"), "rb") as file:
        atac = pickle.load(file)
    if not rna.index.equals(atac.index):
        raise RuntimeError("The two PBMC tables list different cells.")
    save(torch.tensor(rna.to_numpy(), dtype=torch.float32), "PBMC", "rna", "pca50")
    save(torch.tensor(atac.to_numpy(), dtype=torch.float32), "PBMC", "atac", "topics50")
    save_labels(label_indices(pbmc_cell_types().loc[rna.index].tolist()), "PBMC")


def prepare_snare_seq() -> None:
    rna = np.load(raw_file("SNARE/SNAREseq_rna_feat.npy"))  # float64 [cells, 10]
    atac = np.load(raw_file("SNARE/SNAREseq_atac_feat.npy"))  # float64 [cells, 19]
    save(torch.tensor(rna, dtype=torch.float32), "SNARE-seq", "rna", "pca10")
    save(torch.tensor(atac, dtype=torch.float32), "SNARE-seq", "atac", "lsi19")
    cell_types = raw_file("SNARE/SNAREseq_rna_types.txt").read_text().split()  # the same cells in the ATAC file
    save_labels(label_indices(cell_types), "SNARE-seq")


def prepare_cite_seq() -> None:
    files = {"rna": "citeseq_rna_normalizedFC_1000cells.csv", "adt": "citeseq_adt_normalized_1000cells.csv"}
    for modality, name in files.items():
        features_by_cells = pd.read_csv(raw_file(f"CITEseq/{name}"), header=None).to_numpy()  # [25, 1000]
        save(torch.tensor(features_by_cells.T, dtype=torch.float32), "CITE-seq", modality, "raw25")


if __name__ == "__main__":
    prepare_pbmc()
    prepare_snare_seq()
    prepare_cite_seq()
