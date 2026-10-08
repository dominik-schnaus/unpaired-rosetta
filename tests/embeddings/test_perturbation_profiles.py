"""Cell Painting and L1000 (Rosetta). Recompute both files from the replicate profiles and compare them with the stored
ones.

    pixi run -e modalities pytest tests/embeddings/test_perturbation_profiles.py -s

The consensus profiles need all replicates, because the reproducibility filter compares each unit against a null. The
test therefore runs the whole chain: consensus profiles, PCA fit, and projection of all 2,732 rows in the batch sizes
of ``embed.py``. It is deterministic. On an RTX 4500 Ada the result equals the stored files bit for bit. Elsewhere
the last bits may differ.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from torch.nn.functional import cosine_similarity

from modalities import perturbation_profiles
from modalities.pca import fit_pca, project_rows
from modalities.perturbation_profiles import embed, profiles

if TYPE_CHECKING:
    import numpy as np

    from modalities.common import EmbeddingFile

pytestmark = [pytest.mark.embeddings, pytest.mark.slow]

MIN_COSINE = 0.999
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def consensus() -> tuple[dict[str, np.ndarray], list[dict]]:
    if not all(path.exists() for path in perturbation_profiles.REPLICATES.values()):
        pytest.skip("run download.py first")
    return profiles.consensus_profiles()


@pytest.mark.parametrize("file", perturbation_profiles.FILES, ids=lambda file: file.model)
def test_embedding(
    consensus: tuple[dict[str, np.ndarray], list[dict]],
    file: EmbeddingFile,
) -> None:
    if not file.exists():
        pytest.skip(f"missing {file.path}")
    matrix = consensus[0][file.modality]
    computed = project_rows(matrix, fit_pca(matrix, file.shape[1]), embed.BATCH_SIZES[file.modality], DEVICE)
    stored = torch.load(file.path)
    cosines = cosine_similarity(computed, stored, dim=-1)
    numbers = {
        "file": file.relative_path,
        "units": len(consensus[1]),
        "rows": computed.shape[0],
        "exact": bool(torch.equal(computed, stored)),
        "max_abs_difference": (computed - stored).abs().max().item(),
        "min_cosine": cosines.min().item(),
    }
    print("RESULT", json.dumps(numbers))
    assert computed.shape == stored.shape
    assert numbers["min_cosine"] >= MIN_COSINE
