"""Compare our geometry scores with the platonic-rep code (github.com/minyoungg/platonic-rep, ``metrics.py``) on random
inputs. Wang et al. (2026) reuse its CKA with clipping for generative token pooling.

The clipped CKA of App. D differs from a default call of their code in three ways:

* Their ``compute_alignment`` clips with ``exact=True`` by default, which takes one 95th percentile of all entries.
  The paper uses ``exact=False``, the mean over the samples of their 95th percentile. ``clip_outliers`` equals it
  bit for bit.
* Their ``cka`` is the biased estimator (HSIC with the centering matrix). The paper uses ``unbiased_cka``.
* Their CKA divides by ``sqrt(HSIC(K, K) HSIC(L, L)) + 1e-6``, and ``prepare_features`` casts to single precision. We
  use no epsilon and double precision. The two differ by about 1e-5 on unit-norm features. They agree to 1e-12 when
  their functions get double precision and the epsilon is left out.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from torch.nn.functional import normalize

from unpaired_rosetta.geometry import cka, clip_outliers, clipped_cka, mutual_knn

if TYPE_CHECKING:
    from types import ModuleType

OFFICIAL = Path(__file__).parent / "official"
pytestmark = [
    pytest.mark.official,
    pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/platonic_rep/clone.sh"),
]


@pytest.fixture(scope="module")
def official() -> ModuleType:
    """Their ``metrics.py``.

    It imports ``torchaudio`` only for the edit-distance metric, so we replace it with a stub.
    """
    sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
    sys.modules.setdefault("torchaudio.functional", types.ModuleType("torchaudio.functional"))
    spec = importlib.util.spec_from_file_location("platonic_rep_metrics", OFFICIAL / "metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def features(
    num_samples: int = 300,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paired, heavy-tailed, uncentered features, so that clipping and centering both matter."""
    generator = torch.Generator().manual_seed(seed)
    samples_x = torch.randn(num_samples, 32, dtype=torch.float64, generator=generator) ** 3 + 0.5
    samples_y = samples_x @ torch.randn(32, 24, dtype=torch.float64, generator=generator) + torch.randn(
        num_samples, 24, dtype=torch.float64, generator=generator
    )
    return samples_x, samples_y


def test_clipping_equals_remove_outliers(
    official: ModuleType,
) -> None:
    samples_x, _ = features()
    for dtype in [torch.float32, torch.float64]:
        samples = samples_x.to(dtype)
        assert torch.equal(clip_outliers(samples), official.remove_outliers(samples, q=0.95, exact=False))


def test_unbiased_cka_equals_theirs(
    official: ModuleType,
) -> None:
    samples_x, samples_y = features()
    # CKA does not depend on scale. At this scale their epsilon of 1e-6 is far below double precision.
    theirs = official.AlignmentMetrics.measure("unbiased_cka", 100 * samples_x, 100 * samples_y)
    assert cka(samples_x, samples_y) == pytest.approx(theirs, rel=1e-12)


def test_clipped_cka_equals_their_pipeline(
    official: ModuleType,
) -> None:
    samples_x, samples_y = features()
    unit_x = normalize(official.remove_outliers(samples_x, q=0.95, exact=False), p=2, dim=-1)
    unit_y = normalize(official.remove_outliers(samples_y, q=0.95, exact=False), p=2, dim=-1)
    kernel_x, kernel_y = unit_x @ unit_x.T, unit_y @ unit_y.T
    without_epsilon = official.hsic_unbiased(kernel_x, kernel_y) / torch.sqrt(
        official.hsic_unbiased(kernel_x, kernel_x) * official.hsic_unbiased(kernel_y, kernel_y)
    )
    assert clipped_cka(samples_x, samples_y) == pytest.approx(without_epsilon.item(), rel=1e-12)

    # Their default precision and epsilon, as in prepare_features without the move to the GPU.
    single_x = normalize(official.remove_outliers(samples_x.float(), q=0.95, exact=False), p=2, dim=-1)
    single_y = normalize(official.remove_outliers(samples_y.float(), q=0.95, exact=False), p=2, dim=-1)
    assert clipped_cka(samples_x, samples_y) == pytest.approx(
        official.AlignmentMetrics.measure("unbiased_cka", single_x, single_y), abs=1e-4
    )


def test_mutual_knn_equals_theirs(
    official: ModuleType,
) -> None:
    samples_x, samples_y = features()
    unit_x, unit_y = normalize(samples_x, dim=-1), normalize(samples_y, dim=-1)
    for num_neighbors in [1, 10, 30]:
        theirs = official.AlignmentMetrics.measure("mutual_knn", unit_x, unit_y, topk=num_neighbors)
        assert mutual_knn(unit_x, unit_y, num_neighbors) == pytest.approx(theirs, abs=1e-7)
