"""Compare our Local CKA with the official code of Maniparambil et al. (2024)
(github.com/mayug/0-shot-llm-vision, ``src/utils.py``).

``linear_local_CKA`` scores each query and key pair by the linear CKA of the base pairs with the candidate pair added
in front, using explicit centering matrices. ``stretch_representations`` divides each feature by its standard
deviation.

Run ``tests/local_cka/clone.sh`` first.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from unpaired_rosetta.baselines.local_cka import LocalCKA
from unpaired_rosetta.randomness import Randomness

if TYPE_CHECKING:
    from types import ModuleType

OFFICIAL = Path(__file__).parent / "official"
pytestmark = [pytest.mark.official, pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/local_cka/clone.sh")]


@pytest.fixture(scope="module")
def utils() -> ModuleType:
    spec = importlib.util.spec_from_file_location("local_cka_utils", OFFICIAL / "src" / "utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def related_spaces(
    num_samples: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two spaces of different widths that are noisy linear images of the same latent variables."""
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(num_samples, 6, generator=generator)
    x = latent @ torch.randn(6, 10, generator=generator) + 0.3 * torch.randn(num_samples, 10, generator=generator)
    y = latent @ torch.randn(6, 14, generator=generator) + 0.3 * torch.randn(num_samples, 14, generator=generator)
    return x + 2.0, y - 1.0  # not centered, because the stretching does not center


@pytest.mark.parametrize("num_pairs", [1, 2, 10, 30])
def test_similarity_equals_official_linear_local_cka(
    utils: ModuleType,
    num_pairs: int,
) -> None:
    x, y = related_spaces(200 + num_pairs + 14)
    samples_x, samples_y = x[:100], y[100:200]  # unpaired training sets
    paired_x, paired_y = x[200 : 200 + num_pairs], y[200 : 200 + num_pairs]
    queries, keys = x[200 + num_pairs : 200 + num_pairs + 6], y[200 + num_pairs + 6 :]
    aligner = LocalCKA().fit(samples_x, samples_y, paired_x, paired_y, Randomness(0))
    theirs = utils.linear_local_CKA(
        paired_x / aligner.std_x, paired_y / aligner.std_y, queries / aligner.std_x, keys / aligner.std_y, "cpu"
    )
    ours = aligner.similarity(queries, keys)
    assert ours.shape == theirs.shape == (6, 8)
    torch.testing.assert_close(ours, theirs, atol=2e-5, rtol=1e-4)


def test_true_partners_score_highest_far_more_often_than_by_chance() -> None:
    x, y = related_spaces(400)
    aligner = LocalCKA().fit(x[:150], y[150:300], x[300:350], y[300:350], Randomness(0))
    scores = aligner.similarity(x[350:], y[350:])
    assert (scores.argmax(dim=1) == torch.arange(50)).float().mean() > 0.2  # chance: 0.02


def test_stretching_equals_official_on_the_same_samples(
    utils: ModuleType,
) -> None:
    """On the same samples, our stretching equals the official one, ``diag(1 / std)`` up to our floor of 1e-10.

    We divide by the standard deviation of the unpaired training set. The official code divides by the standard
    deviation of the samples it stretches.
    """
    samples_x, samples_y = related_spaces(300)
    aligner = LocalCKA().fit(samples_x, samples_y, None, None, Randomness(0))
    torch.testing.assert_close(samples_x / aligner.std_x, utils.stretch_representations(samples_x))
    torch.testing.assert_close(samples_y / aligner.std_y, utils.stretch_representations(samples_y))


def test_without_pairs_the_similarity_is_constant() -> None:
    samples_x, samples_y = related_spaces(40)
    aligner = LocalCKA().fit(samples_x, samples_y, None, None, Randomness(0))
    assert (aligner.similarity(samples_x, samples_y) == 0).all()
