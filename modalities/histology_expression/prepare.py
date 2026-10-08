"""Join the tiles and the counts of the CCRCC slides on the spot barcode and cache them as two row-aligned arrays.

* ``tiles.npy``: ``[74220, 224, 224, 3]`` uint8, the tile of every spot that has counts.
* ``expression.npy``: ``[74220, 2000]`` float32, log1p counts per 10,000 of the 2,000 most variable genes.

The slides carry different gene annotations, so their gene sets are intersected first. The counts of a spot are
normalized over these shared genes. Among the genes detected in at least 10 % of the spots of the cohort, the 2,000
with the largest variance of the normalized counts are kept. The statistics are summed over the nonzero entries of
one slide at a time (zeros add nothing to either sum), so the full matrix is never in memory. Rows follow the sorted
slide ids, and within a slide the order of the patch file.

    pixi run -e modalities python -m modalities.histology_expression.prepare
"""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from modalities.histology_expression import CACHE, RAW

NUM_GENES = 2000
MIN_DETECTION_RATE = 0.10
COUNTS_PER_SPOT = 1e4


@dataclass
class Slide:
    """The spots of one slide that have both a tile and counts, and where to find them."""

    name: str
    patches_path: Path
    adata_path: Path
    gene_columns: np.ndarray  # columns of the slide's counts in the order of the shared genes
    tile_rows: np.ndarray  # increasing rows of the patch file
    count_rows: np.ndarray  # the matching rows of the counts
    barcodes: list[str]


def slide_names() -> list[str]:
    return sorted(path.stem for path in (RAW / "patches").glob("*.h5"))


def decode(
    values: np.ndarray,
) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in np.asarray(values).ravel()]


def read_counts(
    path: Path,
) -> tuple[csr_matrix, list[str], list[str]]:
    """Raw counts (CSR), spot barcodes and gene symbols of an h5ad file."""
    with h5py.File(path, "r") as handle:
        group = handle["X"]
        counts = csr_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]), shape=tuple(group.attrs["shape"])
        )
        barcodes = decode(handle["obs"]["_index"][:])
        genes = decode(handle["var"]["_index"][:])
    return counts, barcodes, genes


def log1p_cpm(
    counts: csr_matrix,
) -> tuple[csr_matrix, np.ndarray]:
    """log1p of the counts per 10,000 of every row, and the row totals."""
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scaled = counts.multiply((COUNTS_PER_SPOT / np.maximum(totals, 1))[:, None]).tocsr()
    scaled.data = np.log1p(scaled.data)
    return scaled, totals


def shared_genes(
    names: list[str],
) -> list[str]:
    """The genes annotated on every slide, sorted."""
    gene_sets = []
    for name in names:
        with h5py.File(RAW / "adata" / f"{name}.h5ad", "r") as handle:
            gene_sets.append(set(decode(handle["var"]["_index"][:])))
    return sorted(set.intersection(*gene_sets))


def join(
    name: str,
    genes: list[str],
) -> Slide | None:
    """The spots of a slide present in both files, joined on the barcode (the files are not in the same order)."""
    patches_path, adata_path = RAW / "patches" / f"{name}.h5", RAW / "adata" / f"{name}.h5ad"
    _, count_barcodes, slide_genes = read_counts(adata_path)
    with h5py.File(patches_path, "r") as handle:
        tile_barcodes = decode(handle["barcode"][:])
    column_of = {gene: column for column, gene in enumerate(slide_genes)}
    row_of = {barcode: row for row, barcode in enumerate(count_barcodes)}
    rows = [(tile, row_of[barcode]) for tile, barcode in enumerate(tile_barcodes) if barcode in row_of]
    if not rows:
        return None
    return Slide(
        name,
        patches_path,
        adata_path,
        gene_columns=np.array([column_of[gene] for gene in genes]),
        tile_rows=np.array([tile for tile, _ in rows]),
        count_rows=np.array([row for _, row in rows]),
        barcodes=[tile_barcodes[tile] for tile, _ in rows],
    )


def normalized_counts(
    slide: Slide,
) -> tuple[csr_matrix, np.ndarray]:
    counts, _, _ = read_counts(slide.adata_path)
    return log1p_cpm(counts[slide.count_rows][:, slide.gene_columns])


def highly_variable_genes(
    slides: list[Slide],
    num_genes: int,
) -> np.ndarray:
    """Sorted columns of the ``NUM_GENES`` most variable shared genes among those detected in 10 % of the spots."""
    total = np.zeros(num_genes, dtype=np.float64)
    squares = np.zeros(num_genes, dtype=np.float64)
    nonzero = np.zeros(num_genes, dtype=np.int64)
    for slide in slides:
        normalized, _ = normalized_counts(slide)
        total += np.bincount(normalized.indices, weights=normalized.data, minlength=num_genes)
        squares += np.bincount(normalized.indices, weights=normalized.data**2, minlength=num_genes)
        nonzero += np.bincount(normalized.indices, minlength=num_genes)
    num_spots = sum(len(slide.tile_rows) for slide in slides)
    mean = total / num_spots
    variance = squares / num_spots - mean**2
    eligible = np.flatnonzero(nonzero / num_spots >= MIN_DETECTION_RATE)
    selected = eligible[np.argsort(-variance[eligible])[:NUM_GENES]]
    selected.sort()
    return selected


def expression_rows(
    slide: Slide,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The normalized counts of the selected genes (float32) and the count totals of the slide's spots."""
    normalized, totals = normalized_counts(slide)
    return normalized[:, selected].toarray().astype(np.float32), totals


def tile_rows(
    slide: Slide,
) -> np.ndarray:
    with h5py.File(slide.patches_path, "r") as handle:
        return handle["img"][slide.tile_rows]


def main() -> None:
    names = slide_names()
    genes = shared_genes(names)
    slides = [slide for slide in (join(name, genes) for name in names) if slide is not None]
    selected = highly_variable_genes(slides, len(genes))
    num_spots = sum(len(slide.tile_rows) for slide in slides)
    print(f"{num_spots} spots of {len(slides)} slides, {len(genes)} shared genes, {len(selected)} selected")

    CACHE.mkdir(parents=True, exist_ok=True)
    tiles = np.lib.format.open_memmap(CACHE / "tiles.npy", mode="w+", dtype=np.uint8, shape=(num_spots, 224, 224, 3))
    expression = np.zeros((num_spots, len(selected)), dtype=np.float32)
    index, cursor = [], 0
    for slide in slides:
        block, totals = expression_rows(slide, selected)
        tiles[cursor : cursor + len(block)] = tile_rows(slide)
        expression[cursor : cursor + len(block)] = block
        index += [
            {"slide": slide.name, "barcode": barcode, "total_counts": float(np.log1p(total))}
            for barcode, total in zip(slide.barcodes, totals)
        ]
        cursor += len(block)
    tiles.flush()
    np.save(CACHE / "expression.npy", expression)
    (CACHE / "index.json").write_text(json.dumps(index))
    (CACHE / "genes.json").write_text(json.dumps([genes[column] for column in selected]))


if __name__ == "__main__":
    main()
