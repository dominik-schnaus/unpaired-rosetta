"""Human <-> mouse scRNA from the CZ CELLxGENE Census, embedded per species without any shared gene information.

The Census harmonizes cell types to Cell Ontology terms. The 138 terms with at least 300 primary 10x cells in both
species give 250 randomly drawn cells per term and species (34,500 per side). Each species is normalized on its own
(library size 10^4, log1p), reduced to its 2000 most variable genes and embedded by a PCA with 50 components, so
only the geometry connects the two sides. The cell types are stored for evaluation only
(``labels_human.pt`` / ``labels_mouse.pt`` index the sorted shared terms).

Needs the ``cellxgene`` environment (the Census has no wheels for the Python of the default one)::

    pixi run -e cellxgene python -m modalities.single_cell.cross_species
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import torch

from unpaired_rosetta.embeddings import embedding_path, storage_root

if TYPE_CHECKING:
    import pandas as pd
    from tiledbsoma import Collection

DATASET = "CrossSpecies-SC"
CENSUS_VERSION = "2023-12-15"
ORGANISMS = {"human": "homo_sapiens", "mouse": "mus_musculus"}
MIN_CELLS_PER_TYPE = 300
CELLS_PER_TYPE = 250
NUM_GENES = 2000
NUM_COMPONENTS = 50
SEED = 0


def primary_10x_cells(
    census: Collection,
    organism: str,
) -> pd.DataFrame:
    """Cell metadata of the primary (not duplicated) cells measured with a 10x assay."""
    columns = ["soma_joinid", "cell_type", "cell_type_ontology_term_id", "assay", "is_primary_data"]
    cells = census["census_data"][organism].obs.read(column_names=columns).concat().to_pandas()
    return cells[cells["is_primary_data"] & cells["assay"].str.contains("10x", case=False, na=False)]


def shared_cell_types(
    cells: dict[str, pd.DataFrame],
) -> list[str]:
    """The sorted cell-type terms with at least ``MIN_CELLS_PER_TYPE`` cells in every species."""
    counts = [species_cells["cell_type_ontology_term_id"].value_counts() for species_cells in cells.values()]
    return sorted(set.intersection(*(set(count[count >= MIN_CELLS_PER_TYPE].index) for count in counts)))


def sample_cells(
    cells: pd.DataFrame,
    cell_types: list[str],
    generator: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """``CELLS_PER_TYPE`` random cells of every type (sorted ids within a type) and the type of every drawn cell."""
    ids, labels = [], []
    for cell_type in cell_types:
        candidates = cells.loc[cells["cell_type_ontology_term_id"] == cell_type, "soma_joinid"].to_numpy()
        if len(candidates) > CELLS_PER_TYPE:
            candidates = generator.choice(candidates, CELLS_PER_TYPE, replace=False)
        drawn = candidates
        ids.append(np.sort(drawn))
        labels += [cell_type] * len(drawn)
    return np.concatenate(ids), labels


def raw_counts(
    census: Collection,
    organism: str,
    ids: np.ndarray,
) -> sp.csr_matrix:
    """Raw counts ``[cells, genes]`` of the given cells, in the given order."""
    import cellxgene_census

    adata = cellxgene_census.get_anndata(
        census=census, organism=organism, obs_coords=ids.tolist(), column_names={"obs": ["soma_joinid"]}
    )
    row_of = {int(soma_id): row for row, soma_id in enumerate(adata.obs["soma_joinid"].to_numpy())}
    return sp.csr_matrix(adata.X)[[row_of[int(soma_id)] for soma_id in ids]]


def embed(
    counts: sp.csr_matrix,
) -> torch.Tensor:
    """Library-size normalization, log1p, the most variable genes, and a PCA fitted on these cells.

    The PCA runs on a fixed number of BLAS threads because the last bits of its float32 sums depend on the thread
    count. The thread count of the stored files was not recorded, so a recomputation matches them to about 1e-3.
    """
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    library_sizes = np.asarray(counts.sum(1)).ravel()
    library_sizes[library_sizes == 0] = 1
    normalized = counts.multiply((1e4 / library_sizes)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    mean = np.asarray(normalized.mean(0)).ravel()
    variance = np.maximum(np.asarray(normalized.multiply(normalized).mean(0)).ravel() - mean**2, 0)
    variable_genes = np.argsort(-variance)[:NUM_GENES]
    dense = np.asarray(normalized[:, variable_genes].todense(), dtype=np.float32)
    dense -= dense.mean(0, keepdims=True)
    with threadpool_limits(4):
        components = PCA(NUM_COMPONENTS, random_state=SEED).fit_transform(dense)
    return torch.tensor(components, dtype=torch.float32)


def main() -> None:
    import cellxgene_census

    generator = np.random.default_rng(SEED)
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        cells = {species: primary_10x_cells(census, organism) for species, organism in ORGANISMS.items()}
        cell_types = shared_cell_types(cells)
        print(f"{len(cell_types)} cell types with at least {MIN_CELLS_PER_TYPE} cells in both species")
        for species, organism in ORGANISMS.items():  # human first, since both species draw from one generator
            ids, labels = sample_cells(cells[species], cell_types, generator)
            counts = raw_counts(census, organism, ids)
            raw = storage_root() / "raw" / "cross_species_sc" / f"{organism}_counts.npz"
            raw.parent.mkdir(parents=True, exist_ok=True)
            sp.save_npz(raw, counts)  # lets the embedding be recomputed without the Census
            path = embedding_path(DATASET, species, f"pca{NUM_COMPONENTS}")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embed(counts), path)
            labels = torch.tensor([cell_types.index(label) for label in labels])
            torch.save(labels, storage_root() / "embeddings" / DATASET / f"labels_{species}.pt")
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
