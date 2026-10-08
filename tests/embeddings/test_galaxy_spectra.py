"""Galaxy image and spectrum (AstroCLIP DESI). Recompute the cache and the embeddings of the first galaxies and
compare them with the stored ones.

    pixi run pytest tests/embeddings/test_galaxy_spectra.py -s

* ``test_cache_from_download`` streams the first galaxies of the test split (a few hundred MB) and stretches their
  images. Images, spectra, target ids and redshifts must equal the cache exactly.
* ``test_embeddings`` embeds the cached inputs of the first galaxies. The stored files were computed on another GPU,
  so the rows must reach a cosine similarity of 0.999, not bit equality.
* ``test_redshift_deciles`` checks for ten groups of equal size with increasing redshift.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch.nn.functional import cosine_similarity

from modalities import galaxy_spectra
from modalities.galaxy_spectra import download, embed

if TYPE_CHECKING:
    from modalities.common import EmbeddingFile

pytestmark = pytest.mark.embeddings

NUM_GALAXIES = 16
MIN_COSINE = 0.999
HAS_CACHE = (galaxy_spectra.CACHE / "index.json").exists()
HAS_CHECKPOINTS = (galaxy_spectra.CHECKPOINTS / "specformer.ckpt").exists()


@pytest.mark.slow
@pytest.mark.skipif(not HAS_CACHE, reason="no cached galaxies")
def test_cache_from_download() -> None:
    images, spectra, index = zip(*download.galaxies(limit=4))
    cached_index = json.loads((galaxy_spectra.CACHE / "index.json").read_text())[:4]
    cached_images = np.load(galaxy_spectra.CACHE / "images.npy", mmap_mode="r")[:4]
    cached_spectra = np.load(galaxy_spectra.CACHE / "spectra.npy", mmap_mode="r")[:4]
    print(
        f"RESULT cache: images equal {np.array_equal(np.stack(images), cached_images)}, spectra equal "
        f"{np.array_equal(np.stack(spectra), cached_spectra)}, index equal {list(index) == cached_index}"
    )
    assert np.array_equal(np.stack(images), cached_images)
    assert np.array_equal(np.stack(spectra), cached_spectra)
    assert list(index) == cached_index


@pytest.mark.gpu
@pytest.mark.skipif(
    not HAS_CACHE or not HAS_CHECKPOINTS or not torch.cuda.is_available(),
    reason="needs the cached galaxies, the checkpoints and a GPU",
)
@pytest.mark.parametrize("file", galaxy_spectra.FILES, ids=lambda file: file.model)
def test_embeddings(
    file: EmbeddingFile,
) -> None:
    if file is galaxy_spectra.IMAGE:
        model = embed.load_astrodino(galaxy_spectra.CHECKPOINTS / "astrodino.ckpt").cuda()
        images = np.load(galaxy_spectra.CACHE / "images.npy", mmap_mode="r")[:NUM_GALAXIES]
        computed = embed.embed_images(model, images)
    else:
        model = embed.load_specformer(galaxy_spectra.CHECKPOINTS / "specformer.ckpt").eval().cuda()
        spectra = np.load(galaxy_spectra.CACHE / "spectra.npy", mmap_mode="r")[:NUM_GALAXIES]
        computed = embed.embed_spectra(model, spectra)
    stored = torch.load(file.path, mmap=True)[:NUM_GALAXIES]
    cosines = cosine_similarity(computed, stored, dim=-1)
    print(
        f"RESULT {file.relative_path}: min cosine {cosines.min().item():.7f}, "
        f"max |diff| {(computed - stored).abs().max().item():.3g}"
    )
    assert cosines.min() > MIN_COSINE


@pytest.mark.skipif(not HAS_CACHE, reason="no cached galaxies")
def test_redshift_deciles() -> None:
    groups, names = galaxy_spectra.redshift_deciles()
    redshifts = torch.as_tensor(galaxy_spectra.redshifts())
    sizes = torch.bincount(groups, minlength=len(names))
    assert groups.shape == (galaxy_spectra.NUM_GALAXIES,) and names == [f"z{group}" for group in range(10)]
    assert sizes.max() - sizes.min() <= 1
    assert all(redshifts[groups == group].max() <= redshifts[groups == group + 1].min() for group in range(9))
