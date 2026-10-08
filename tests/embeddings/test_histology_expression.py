"""Histology and expression (HEST CCRCC). Recompute the cache of the first slide and the embeddings and compare them.

    pixi run -e modalities pytest tests/embeddings/test_histology_expression.py -s

* ``prepare.py``: the tiles and the expression rows of the first slide must equal the cached arrays exactly. The gene
  selection needs all 24 slides.
* ``expression-pca64-ccrcc``: the PCA of the whole cached expression matrix, projected in the batch size of
  ``embed.py``. On an RTX 4500 Ada it equals the stored file bit for bit. Elsewhere the last bits may differ.
* ``h_optimus_0``: the first tiles, in float32 on a GPU, must reach a cosine similarity above 0.999 with the stored
  rows.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch.nn.functional import cosine_similarity

from modalities.histology_expression import CACHE, EXPRESSION, RAW, TILE, embed, prepare

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.embeddings

NUM_TILES = 16
MIN_COSINE = 0.999
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def require(
    *paths: Path,
) -> None:
    for path in paths:
        if not path.exists():
            pytest.skip(f"{path} is missing")


@pytest.mark.slow
def test_first_slide_cache() -> None:
    require(RAW / "patches", CACHE / "tiles.npy", CACHE / "expression.npy")
    names = prepare.slide_names()
    genes = prepare.shared_genes(names)
    slides = [slide for slide in (prepare.join(name, genes) for name in names) if slide is not None]
    selected = prepare.highly_variable_genes(slides, len(genes))
    expression, _ = prepare.expression_rows(slides[0], selected)
    tiles = prepare.tile_rows(slides[0])
    cached_tiles = np.load(CACHE / "tiles.npy", mmap_mode="r")[: len(tiles)]
    cached_expression = np.load(CACHE / "expression.npy", mmap_mode="r")[: len(expression)]
    print(
        "RESULT",
        json.dumps(
            {
                "slide": slides[0].name,
                "spots": len(tiles),
                "tiles_equal": bool(np.array_equal(tiles, cached_tiles)),
                "expression_max_abs_difference": float(np.abs(expression - cached_expression).max()),
            }
        ),
    )
    np.testing.assert_array_equal(tiles, cached_tiles)
    np.testing.assert_array_equal(expression, cached_expression)


def test_expression_embedding() -> None:
    require(CACHE / "expression.npy", EXPRESSION.path)
    computed = embed.embed_expression(DEVICE)
    stored = torch.load(EXPRESSION.path)
    cosines = cosine_similarity(computed, stored, dim=-1)
    print(
        "RESULT",
        json.dumps(
            {
                "file": EXPRESSION.relative_path,
                "rows": computed.shape[0],
                "exact": bool(torch.equal(computed, stored)),
                "max_abs_difference": (computed - stored).abs().max().item(),
                "min_cosine": cosines.min().item(),
            }
        ),
    )
    assert cosines.min().item() > MIN_COSINE


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_tile_embedding() -> None:
    require(CACHE / "tiles.npy", TILE.path)
    tiles = np.load(CACHE / "tiles.npy", mmap_mode="r")[:NUM_TILES]
    computed = embed.embed_tiles(embed.h_optimus_0("cuda"), tiles)
    stored = torch.load(TILE.path, mmap=True)[:NUM_TILES]
    cosines = cosine_similarity(computed, stored, dim=-1)
    print(
        f"RESULT {TILE.relative_path}: min cosine {cosines.min().item():.7f}, max |diff| {(computed - stored).abs().max().item():.3g}"
    )
    assert cosines.min().item() > MIN_COSINE
