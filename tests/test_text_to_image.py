"""Tests of the text-to-image experiment code. They cover the grid of fits, the language-to-image maps, the rows and
shards of the benchmark, the noise seeds and the aggregation of Tables 6 and 7."""

import math

import pandas as pd
import pytest
import torch

from experiments import text_to_image, text_to_image_scores
from modalities.text_to_image import prompt_seed
from unpaired_rosetta.baselines.linear import LinearMap
from unpaired_rosetta.randomness import Randomness
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes


def test_fits_of_the_grids() -> None:
    configurations = text_to_image.configurations()
    assert len(configurations) == 2 * 2 * 8 and len(set(configurations)) == len(configurations)
    assert {c.seed for c in configurations} == {734796314}
    assert all(
        c.holdout == 8192 and c.validation == "coco_train2014" and c.model_x == "rae_dinov2_base_mean"
        for c in configurations
    )


def paired_data(
    num_samples: int = 300,
    dimension: int = 6,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    samples_x = torch.randn(num_samples, dimension, generator=generator)
    rotation = torch.linalg.qr(torch.randn(dimension, dimension, generator=generator))[0]
    return samples_x, samples_x @ rotation + 0.01 * torch.randn(num_samples, dimension, generator=generator)


def test_our_map_inverts_the_fitted_rotation() -> None:
    """The transpose of our semi-orthogonal map takes a text embedding next to its image embedding."""
    samples_x, samples_y = paired_data()
    aligner = WassersteinProcrustes(num_clusters=4, num_restarts=2, batch_size=300, num_iterations=3)
    aligner.fit(samples_x, samples_y, samples_x[:50], samples_y[:50], Randomness(0))
    config = text_to_image.configurations()[0]
    language_map = text_to_image.language_to_image_map(aligner, config, Randomness(0))
    assert torch.equal(language_map["text_to_image"], aligner.weight.T)
    mapped = (
        (samples_y - aligner.mean_y)
        / (samples_y - aligner.mean_y).norm(dim=1, keepdim=True)
        @ language_map["text_to_image"]
    )
    target = (samples_x - aligner.mean_x) / (samples_x - aligner.mean_x).norm(dim=1, keepdim=True)
    assert (mapped * target).sum(dim=1).mean() > 0.99


def test_linear_map_without_pairs_is_a_gaussian_matrix_of_the_run() -> None:
    """Without pairs, there is nothing to fit, so the linear map is a Gaussian matrix drawn right after the split."""
    samples_x, samples_y = paired_data()
    aligner = LinearMap().fit(samples_x, samples_y, None, None, Randomness(3))
    config = next(c for c in text_to_image.configurations() if c.method == "linear" and c.num_pairs == 0)
    randomness = Randomness(3)
    language_map = text_to_image.language_to_image_map(aligner, config, randomness)
    expected = torch.randn(6, 6, generator=Randomness(3).torch)
    assert torch.allclose(language_map["text_to_image"], torch.linalg.pinv(expected))


def test_benchmark_rows_and_shards() -> None:
    rows = text_to_image_scores.rows()
    assert rows[:2] == ["gt", "scale-rae"] and len(rows) == 18 and len(set(rows)) == 18
    shards = text_to_image_scores.configurations()
    for dataset, count in (("cycleprefdb", 380), ("coco_val2014", 40504)):
        covered = [i for shard in shards if shard.dataset == dataset for i in range(shard.start, shard.stop)]
        assert covered == list(range(count))
    assert text_to_image_scores.row_labels("linear_contriever_100") == ("Contriever", "linear", "100")


def test_prompt_seeds_are_fixed_and_distinct() -> None:
    seeds = [prompt_seed(i) for i in range(10_000)]
    assert len(set(seeds)) == len(seeds) and all(0 <= seed < 2**63 for seed in seeds)
    assert prompt_seed(0) == int.from_bytes(__import__("hashlib").sha256(b"734796314:0").digest()[:8], "little") % 2**63


def test_mean_and_standard_error_leave_out_prompts_without_questions() -> None:
    scores = pd.DataFrame(
        {"row": ["gt"] * 4, "index": range(4), "metric": ["tifa"] * 4, "score": [1.0, 0.5, math.nan, 0.0]}
    )
    summary = text_to_image_scores.summary(scores).iloc[0]
    assert summary["count"] == 3 and summary["mean"] == pytest.approx(0.5)
    assert summary["standard_error"] == pytest.approx(0.5 / math.sqrt(3))
