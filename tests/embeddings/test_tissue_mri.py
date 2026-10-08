"""Tissue and MRI (TCGA glioma). Recompute the embeddings of the first patients and compare them with the stored rows.

    pixi run -e modalities pytest tests/embeddings/test_tissue_mri.py -s

* MRI: the composites are cut from the BraTS zip again and embedded with DINO ViT-B/16. The rows must reach a cosine
  similarity of 0.999 with the stored ones (float32 on another GPU).
* Pathology: the stored slide embeddings drew their 256 patches with an unseeded random permutation, and the slides
  were deleted afterwards. No row can therefore be recomputed exactly. ``test_pathology_embedding`` downloads the
  slide of one patient again (0.5 GB), draws a new seeded set of patches, and checks that the mean embedding still
  agrees with the stored row up to the patch sampling (cosine similarity above ``MIN_RESAMPLED_COSINE``).
"""

from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING

import pytest
import torch
from torch.nn.functional import cosine_similarity

from modalities import tissue_mri
from modalities.tissue_mri import embed

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.embeddings

NUM_PATIENTS = 4
MIN_COSINE = 0.999
MIN_RESAMPLED_COSINE = 0.99
HAS_BRATS = tissue_mri.BRATS_ZIP.exists()


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_BRATS or not torch.cuda.is_available(), reason="needs BRATS-2020.zip and a GPU")
def test_mri_embedding() -> None:
    model = embed.dino_vitb16("cuda")
    with zipfile.ZipFile(tissue_mri.BRATS_ZIP) as archive:
        subjects = tissue_mri.cohort()["brats"][:NUM_PATIENTS]
        computed = torch.stack([embed.embed_mri(model, embed.mri_composites(archive, subject)) for subject in subjects])
    stored = torch.load(tissue_mri.MRI.path, mmap=True)[:NUM_PATIENTS]
    cosines = cosine_similarity(computed, stored, dim=-1)
    print(
        f"RESULT {tissue_mri.MRI.relative_path}: min cosine {cosines.min().item():.6f}, "
        f"max |diff| {(computed - stored).abs().max().item():.3g}"
    )
    assert cosines.min() > MIN_COSINE


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_pathology_embedding(
    tmp_path: Path,
) -> None:
    patient = tissue_mri.cohort().iloc[1]  # the smallest of the first slides (0.48 GB)
    model = embed.gigapath("cuda")
    computed = embed.embed_slide(model, patient["gdc_file"], tmp_path)
    stored = torch.load(tissue_mri.PATHOLOGY.path, mmap=True)[1]
    cosine = cosine_similarity(computed, stored, dim=0).item()
    print(f"RESULT {tissue_mri.PATHOLOGY.relative_path} ({patient['barcode']}, new patches): cosine {cosine:.6f}")
    assert cosine > MIN_RESAMPLED_COSINE
