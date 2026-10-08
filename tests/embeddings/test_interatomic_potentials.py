"""Two interatomic potentials (MP-20). Recompute both files from the MACE descriptors and compare them with the
stored ones.

    pixi run pytest tests/embeddings/test_interatomic_potentials.py -s            # the PCA of the cached descriptors
    pixi run -e mlip pytest tests/embeddings/test_interatomic_potentials.py -s    # the MACE forward passes

* ``test_pca`` fits the PCA on the cached descriptors and projects all rows in the batch sizes of ``embed.py``. On an
  RTX 4500 Ada the result equals the stored file bit for bit. On other hardware the last bits may differ.
* ``test_descriptors`` recomputes the descriptors of the first structures with MACE on a GPU, projects them with the
  PCA of the cached descriptors, and compares both with the cached descriptors and the stored rows. Those were
  computed on another GPU, so they must reach a cosine similarity of 0.999, not bit equality.

The ``mlip`` environment cannot import the packages of this repository, so ``descriptors.py`` is loaded from its path.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch.nn.functional import cosine_similarity

from modalities.pca import fit_pca, project

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.embeddings

POTENTIALS = {"mace_mp_small": "small", "mace_mp_medium": "medium"}
NUM_STRUCTURES = 16
MIN_COSINE = 0.999
IN_MLIP_ENVIRONMENT = importlib.util.find_spec("mace") is not None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_descriptors_script() -> ModuleType:
    path = Path(__file__).parents[2] / "modalities" / "interatomic_potentials" / "descriptors.py"
    spec = importlib.util.spec_from_file_location("descriptors", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stored_rows(
    potential: str,
) -> torch.Tensor:
    root = Path(os.environ.get("UNPAIRED_ROSETTA_ROOT", "storage")).expanduser()
    path = root / "embeddings" / "MP20-structure" / potential / f"{potential}-pca256-structure.pt"
    if not path.exists():
        pytest.skip(f"missing {path}")
    return torch.load(path)


def result(
    potential: str,
    what: str,
    computed: torch.Tensor,
    stored: torch.Tensor,
) -> dict:
    cosines = cosine_similarity(computed.float(), stored.float(), dim=-1)
    numbers = {
        "potential": potential,
        "compared": what,
        "rows": computed.shape[0],
        "exact": bool(torch.equal(computed, stored)),
        "max_abs_difference": (computed - stored).abs().max().item(),
        "min_cosine": cosines.min().item(),
    }
    print("RESULT", json.dumps(numbers))
    return numbers


@pytest.mark.skipif(IN_MLIP_ENVIRONMENT, reason="runs in the default environment")
@pytest.mark.parametrize("potential", POTENTIALS)
def test_pca(
    potential: str,
) -> None:
    from modalities import interatomic_potentials
    from modalities.interatomic_potentials import embed

    if not interatomic_potentials.descriptor_path(potential).exists():
        pytest.skip("run descriptors.py first")
    computed = embed.embed(interatomic_potentials.embedding_file(potential), DEVICE)
    assert result(potential, "pca", computed, stored_rows(potential))["min_cosine"] >= MIN_COSINE


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not IN_MLIP_ENVIRONMENT, reason="runs in the mlip environment")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("potential", POTENTIALS)
def test_descriptors(
    potential: str,
) -> None:
    descriptors = load_descriptors_script()
    cache = descriptors.DESCRIPTORS / f"{potential}.npy"
    if not (descriptors.STRUCTURES_CSV.exists() and cache.exists()):
        pytest.skip("run download.py and descriptors.py first")
    structures, _ = descriptors.load_structures(NUM_STRUCTURES)
    computed = torch.from_numpy(descriptors.structure_descriptors(POTENTIALS[potential], structures))
    cached = np.load(cache, mmap_mode="r")
    assert (
        result(potential, "descriptors", computed, torch.from_numpy(cached[:NUM_STRUCTURES].copy()))["min_cosine"]
        >= MIN_COSINE
    )
    basis = fit_pca(cached, 256)
    projected = project(computed, basis)
    assert result(potential, "pca rows", projected, stored_rows(potential)[:NUM_STRUCTURES])["min_cosine"] >= MIN_COSINE
