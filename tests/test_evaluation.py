"""Tests that the evaluation protocol cannot be cheated. They check disjoint training halves, shuffled validation and
the metrics."""

import pytest
import torch

from unpaired_rosetta.evaluation import (
    assert_disjoint,
    disjoint_split,
    independent_split,
    label_transfer_accuracy,
    midranks,
    retrieval_metrics,
    sample_pairs,
    zero_shot_accuracy,
)
from unpaired_rosetta.linalg import polar
from unpaired_rosetta.randomness import Randomness


class ConstantAligner:
    """Gives every pair the same score."""

    def similarity(
        self,
        queries_x: torch.Tensor,
        keys_y: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(queries_x.shape[0], keys_y.shape[0])


class RowOrderAligner:
    """Tries to cheat by assuming that query i belongs to key i in the validation data."""

    def similarity(
        self,
        queries_x: torch.Tensor,
        keys_y: torch.Tensor,
    ) -> torch.Tensor:
        return torch.eye(queries_x.shape[0], keys_y.shape[0])


class MapAligner:
    """Scores ``x W y^T`` for a fixed map ``W``."""

    def __init__(
        self,
        weight: torch.Tensor,
    ) -> None:
        self.weight = weight

    def similarity(
        self,
        queries_x: torch.Tensor,
        keys_y: torch.Tensor,
    ) -> torch.Tensor:
        return (queries_x @ self.weight) @ keys_y.T


def paired_validation(
    num_samples: int = 2000,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    samples_x = torch.nn.functional.normalize(
        torch.randn(num_samples, 8, dtype=torch.float64, generator=generator), dim=1
    )
    rotation = polar(torch.randn(8, 12, dtype=torch.float64, generator=generator))
    samples_y = torch.nn.functional.normalize(
        samples_x @ rotation + 0.3 * torch.randn(num_samples, 12, dtype=torch.float64, generator=generator), dim=1
    )
    return samples_x, samples_y, rotation


@pytest.mark.parametrize("num_samples", [10, 11, 82783])
def test_disjoint_split_gives_two_disjoint_shuffled_halves(
    num_samples: int,
) -> None:
    indices_x, indices_y = disjoint_split(num_samples, Randomness(0))
    assert len(indices_x) == num_samples // 2
    assert len(indices_y) == num_samples - num_samples // 2  # the odd item goes to the second side
    assert not set(indices_x.tolist()) & set(indices_y.tolist())
    assert sorted(indices_x.tolist() + indices_y.tolist()) == list(range(num_samples))
    assert not torch.equal(indices_x, indices_x.sort().values)  # shuffled


def test_disjoint_split_with_a_size_limit() -> None:
    indices_x, indices_y = disjoint_split(1000, Randomness(1), size=100)
    assert len(indices_x) == len(indices_y) == 100
    assert not set(indices_x.tolist()) & set(indices_y.tolist())


def test_overlapping_training_sets_are_rejected() -> None:
    with pytest.raises(AssertionError):
        assert_disjoint(torch.tensor([1, 2, 3]), torch.tensor([3, 4]))


def test_independent_split_and_pairs() -> None:
    indices_x, indices_y = independent_split(100, 50, Randomness(2))
    assert sorted(indices_x.tolist()) == list(range(100)) and sorted(indices_y.tolist()) == list(range(50))
    small, large = sample_pairs(1000, 5, Randomness(3)), sample_pairs(1000, 50, Randomness(3))
    assert torch.equal(small, large[:5])  # nested across numbers of pairs


def foscttm_by_definition(
    mapped_x: torch.Tensor,
    samples_y: torch.Tensor,
) -> float:
    """Eq. (11) of the paper with Euclidean distances. It is the fraction of y_j closer to f(x_i) than y_i, with ties
    counted as one half."""
    distances = torch.cdist(mapped_x, samples_y)
    true_distances = distances.diagonal()[:, None]
    closer = (distances < true_distances).sum(dim=1).double() + 0.5 * (
        (distances == true_distances).sum(dim=1).double() - 1
    )
    return (closer / (len(samples_y) - 1)).mean().item()


def test_foscttm_matches_the_definition_of_the_paper() -> None:
    samples_x, samples_y, rotation = paired_validation()
    metrics = retrieval_metrics(MapAligner(rotation), samples_x, samples_y, Randomness(0), block_size=300)
    # The mapped rows keep unit length, so distances are ordered like negative inner products.
    assert metrics["foscttm"] == pytest.approx(foscttm_by_definition(samples_x @ rotation, samples_y), abs=1e-12)
    assert 0 < metrics["foscttm"] < 0.1


def test_mean_rank_is_one_plus_foscttm_times_n_minus_one() -> None:
    samples_x, samples_y, rotation = paired_validation(500)
    metrics = retrieval_metrics(MapAligner(rotation), samples_x, samples_y, Randomness(0))
    assert metrics["mean_rank"] == pytest.approx(1 + metrics["foscttm"] * 499)


def test_a_constant_aligner_is_exactly_at_chance() -> None:
    samples_x, samples_y, _ = paired_validation(1000)
    metrics = retrieval_metrics(ConstantAligner(), samples_x, samples_y, Randomness(0))
    assert metrics["foscttm"] == 0.5


def test_the_row_order_carries_no_information() -> None:
    samples_x, samples_y, _ = paired_validation(3000)
    foscttms = [
        retrieval_metrics(RowOrderAligner(), samples_x, samples_y, Randomness(seed))["foscttm"] for seed in range(5)
    ]
    for foscttm in foscttms:
        assert foscttm == pytest.approx(0.5, abs=0.01)


def test_an_oracle_map_is_near_perfect() -> None:
    samples_x, samples_y, rotation = paired_validation()
    samples_y = torch.nn.functional.normalize(samples_x @ rotation, dim=1)  # no noise
    assert retrieval_metrics(MapAligner(rotation), samples_x, samples_y, Randomness(0))["foscttm"] == 0.0


def test_metrics_do_not_depend_on_the_shuffle() -> None:
    samples_x, samples_y, rotation = paired_validation()
    results = [
        retrieval_metrics(MapAligner(rotation), samples_x, samples_y, Randomness(seed), block_size=512)
        for seed in range(3)
    ]
    assert results[0] == results[1] == results[2]


def test_midranks_count_ties_half() -> None:
    scores = torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.5, 2.0, 0.5, 0.5]], dtype=torch.float64)
    true_scores = torch.tensor([1.0, 0.5], dtype=torch.float64)
    torch.testing.assert_close(midranks(scores, true_scores), torch.tensor([2.0, 3.0], dtype=torch.float64))


def test_zero_shot_accuracy() -> None:
    class_prompts = torch.eye(6)
    labels = torch.randint(0, 6, (300,), generator=torch.Generator().manual_seed(0))
    images = torch.eye(6)[labels] + 0.5 * torch.eye(6)[(labels + 5) % 6]
    aligner = MapAligner(torch.eye(6))
    assert zero_shot_accuracy(aligner, images, class_prompts, labels, Randomness(0)) == {"top1": 1.0, "top5": 1.0}
    wrong = MapAligner(torch.roll(torch.eye(6), 1, dims=1))  # the true class gets the second highest score
    result = zero_shot_accuracy(wrong, images, class_prompts, labels, Randomness(0))
    assert result["top1"] == 0.0 and result["top5"] == 1.0


def test_label_transfer_accuracy() -> None:
    labels = torch.arange(100) % 4
    samples = torch.eye(4)[labels] + 0.01 * torch.randn(100, 4, generator=torch.Generator().manual_seed(0))
    aligner = MapAligner(torch.eye(4))
    assert label_transfer_accuracy(aligner, samples, samples, labels, labels, Randomness(0)) == 1.0
    swapped = MapAligner(torch.eye(4)[[1, 0, 2, 3]])
    assert label_transfer_accuracy(swapped, samples, samples, labels, labels, Randomness(0)) == 0.5
