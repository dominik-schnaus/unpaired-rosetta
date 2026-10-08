"""Compare our ASIF with the official code (github.com/noranta4/ASIF, ``relreps.py``).

ASIF takes relative representations with respect to the anchors, keeps the top ``k``, raises them to the power 8,
normalizes them and compares them by inner product.

Run ``tests/asif/clone.sh`` first.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from unpaired_rosetta.baselines.asif import ASIF
from unpaired_rosetta.linalg import center_and_normalize
from unpaired_rosetta.randomness import Randomness

if TYPE_CHECKING:
    from types import ModuleType

OFFICIAL = Path(__file__).parent / "official"
pytestmark = [pytest.mark.official, pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/asif/clone.sh")]


@pytest.fixture(scope="module")
def relreps() -> ModuleType:
    spec = importlib.util.spec_from_file_location("asif_relreps", OFFICIAL / "relreps.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_relative_representation(
    relreps: ModuleType,
    samples: torch.Tensor,
    anchors: torch.Tensor,
    non_zeros: int,
    clip: bool = False,
) -> torch.Tensor:
    """The official pipeline of ``zero_shot_classification`` (one anchor chunk, exponent 8) as a dense matrix.

    ``clip`` first sets the negative similarities to zero, the one step our version adds.
    """
    indices, values = relreps.relative_represent(samples, anchors, non_zeros=non_zeros)
    values = values.clamp(min=0) if clip else values
    sparse = relreps.sparsify(indices, values**8, (samples.shape[0], anchors.shape[0]))
    return relreps.normalize_sparse(sparse, non_zeros).to_dense()


def clustered(
    centers: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
    spread: float = 1.0,
) -> torch.Tensor:
    """Samples around the ``centers``.

    With a small ``spread``, each sample has only positive cosines to the anchors of its own cluster. With a large
    ``spread``, many cosines are negative.
    """
    labels = torch.randint(0, centers.shape[0], (num_samples,), generator=generator)
    return centers[labels] + spread * torch.randn(num_samples, centers.shape[1], generator=generator)


def setting(
    num_anchors: int,
    num_neighbors: int,
    spread: float = 1.0,
    seed: int = 0,
) -> tuple[ASIF, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """A fitted ASIF, raw validation queries and keys, and the same queries and keys preprocessed."""
    generator = torch.Generator().manual_seed(seed)
    centers_x, centers_y = 3 * torch.randn(2, 16, generator=generator), 3 * torch.randn(2, 24, generator=generator)
    samples_x, samples_y = clustered(centers_x, 200, generator), clustered(centers_y, 200, generator)
    paired_x = clustered(centers_x, num_anchors, generator, spread)
    paired_y = clustered(centers_y, num_anchors, generator, spread)
    queries, keys = clustered(centers_x, 30, generator, spread), clustered(centers_y, 40, generator, spread)
    aligner = ASIF(num_neighbors=num_neighbors).fit(samples_x, samples_y, paired_x, paired_y, Randomness(seed))
    preprocessed = center_and_normalize(queries, aligner.mean_x), center_and_normalize(keys, aligner.mean_y)
    return aligner, queries, keys, preprocessed


def kept_similarities(
    samples: torch.Tensor,
    anchors: torch.Tensor,
    num_neighbors: int,
) -> torch.Tensor:
    return torch.topk(samples @ anchors.T, num_neighbors, dim=1).values


@pytest.mark.parametrize("num_anchors, num_neighbors", [(60, 10), (200, 50), (10, 3)])
def test_similarity_equals_official_when_every_kept_similarity_is_positive(
    relreps: ModuleType,
    num_anchors: int,
    num_neighbors: int,
) -> None:
    aligner, queries, keys, (queries_preprocessed, keys_preprocessed) = setting(num_anchors, num_neighbors, spread=0.3)
    assert (kept_similarities(queries_preprocessed, aligner.anchors_x, num_neighbors) > 0).all()
    assert (kept_similarities(keys_preprocessed, aligner.anchors_y, num_neighbors) > 0).all()
    relative_x = official_relative_representation(relreps, queries_preprocessed, aligner.anchors_x, num_neighbors)
    relative_y = official_relative_representation(relreps, keys_preprocessed, aligner.anchors_y, num_neighbors)
    torch.testing.assert_close(aligner.similarity(queries, keys), relative_x @ relative_y.T, atol=1e-6, rtol=1e-5)


def test_negative_similarities_are_clipped_before_the_power(
    relreps: ModuleType,
) -> None:
    """With as many neighbors as anchors (the few-pair setting), we clip negative cosines to zero.

    Dissimilar anchors are then kept too. The official code raises their negative cosines to the power 8, which makes
    them count as similar.
    """
    aligner, queries, keys, (queries_preprocessed, keys_preprocessed) = setting(10, 800, spread=3.0)
    assert (queries_preprocessed @ aligner.anchors_x.T < 0).any()
    relative = [
        official_relative_representation(relreps, samples, anchors, 10, clip=clip)
        for samples, anchors in [(queries_preprocessed, aligner.anchors_x), (keys_preprocessed, aligner.anchors_y)]
        for clip in (False, True)
    ]
    unclipped, clipped = relative[0] @ relative[2].T, relative[1] @ relative[3].T
    ours = aligner.similarity(queries, keys)
    torch.testing.assert_close(ours, clipped, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(ours, unclipped, atol=1e-3)


def test_without_pairs_the_similarity_is_constant() -> None:
    generator = torch.Generator().manual_seed(0)
    samples_x, samples_y = torch.randn(20, 4, generator=generator), torch.randn(20, 5, generator=generator)
    aligner = ASIF().fit(samples_x, samples_y, None, None, Randomness(0))
    assert (aligner.similarity(samples_x, samples_y) == 0).all()
