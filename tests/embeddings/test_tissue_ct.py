"""Tissue and CT (CPTAC). Recompute the cached inputs and embeddings of the first patients and compare them with the
stored ones.

    pixi run -e modalities pytest tests/embeddings/test_tissue_ct.py -s

* ``test_cache_from_dicom`` downloads the DICOM of one patient (about 150 MB) from the IDC and rebuilds its cached
  tiles and CT volume, which must equal the cache exactly.
* ``test_embeddings`` embeds the cached inputs of the first patients. The stored files were computed on another GPU
  with bfloat16 autocast, so the rows must reach a cosine similarity of 0.999, not bit equality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
import torch
from torch.nn.functional import cosine_similarity

from modalities import tissue_ct
from modalities.tissue_ct import download, embed

if TYPE_CHECKING:
    from pathlib import Path

    from modalities.common import EmbeddingFile

pytestmark = pytest.mark.embeddings

NUM_PATIENTS = 4
MIN_COSINE = 0.999
DICOM_PATIENT = "C3L-04853"  # the smallest download among the first cached patients (150 MB)
HAS_CACHE = tissue_ct.CACHE.exists()


def cached(
    patient: str,
) -> dict:
    return torch.load(tissue_ct.CACHE / f"{patient}.pt", weights_only=False)


@pytest.mark.slow
@pytest.mark.skipif(not HAS_CACHE, reason="no cached CPTAC inputs")
def test_cache_from_dicom(
    tmp_path: Path,
) -> None:
    from idc_index import index

    series = pd.read_csv(tissue_ct.COHORT).set_index("patient").loc[DICOM_PATIENT]
    client = index.IDCClient()
    tiles, mpp = download.tissue_tiles(download.fetch(client, series.sm_series, tmp_path / "sm"))
    volume = download.ct_volume(download.fetch(client, series.ct_series, tmp_path / "ct"))
    reference = cached(DICOM_PATIENT)
    print(
        f"RESULT tiles {tuple(tiles.shape)} vs {tuple(reference['tiles'].shape)}, mpp {mpp} vs {reference['mpp']}, "
        f"CT max |diff| {(volume.float() - reference['ct'].float()).abs().max().item()}"
    )
    assert torch.equal(tiles, reference["tiles"])
    assert torch.equal(volume, reference["ct"])


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CACHE or not torch.cuda.is_available(), reason="needs the cached inputs and a GPU")
@pytest.mark.parametrize("file", tissue_ct.FILES, ids=lambda file: file.model)
def test_embeddings(
    file: EmbeddingFile,
) -> None:
    build, embed_one, key = {
        "h_optimus_0": (embed.h_optimus_0, embed.embed_slide, "tiles"),
        "ct_fm": (embed.ct_fm, embed.embed_volume, "ct"),
    }[file.model]
    model = build("cuda")
    computed = torch.stack([embed_one(model, cached(patient)[key]) for patient in tissue_ct.patients()[:NUM_PATIENTS]])
    stored = torch.load(file.path, mmap=True)[:NUM_PATIENTS]
    cosines = cosine_similarity(computed, stored, dim=-1)
    print(
        f"RESULT {file.relative_path}: min cosine {cosines.min().item():.6f}, "
        f"max |diff| {(computed - stored).abs().max().item():.3g}"
    )
    assert cosines.min() > MIN_COSINE
