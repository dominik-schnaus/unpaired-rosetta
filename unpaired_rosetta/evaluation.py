"""Data splits and metrics of the evaluation protocol.

An aligner only sees two unpaired training sets that share no item, the ``p`` known pairs in the few-pair setting,
and validation queries and keys that are shuffled separately. Only the evaluator knows the true partners, so a score
that uses the row order gets the chance level FOSCTTM of 0.5.
"""

from typing import Protocol

import torch
from torch import Tensor

from unpaired_rosetta.randomness import Randomness


class Aligner(Protocol):
    """Interface of an aligner (ours or a baseline)."""

    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None,
        paired_y: Tensor | None,
        randomness: Randomness,
    ) -> object: ...

    def similarity(
        self,
        queries_x: Tensor,
        keys_y: Tensor,
    ) -> Tensor:
        """``[num_queries, num_keys]`` scores. Larger means more similar."""


def disjoint_split(
    num_samples: int,
    randomness: Randomness,
    size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Indices of two unpaired training sets of a paired corpus, the two halves of one random permutation.

    Args:
        num_samples: Number of items in the paired corpus.
        randomness: Generators of the run.
        size: Keep only the first ``size`` indices of each half. None keeps all.

    Returns:
        The indices of X and of Y. For odd ``num_samples`` the second half is one longer.
    """
    permutation = randomness.permutation(num_samples)
    indices_x = permutation[: num_samples // 2][:size]
    indices_y = permutation[num_samples // 2 :][:size]
    assert_disjoint(indices_x, indices_y)
    return indices_x, indices_y


def independent_split(
    num_samples_x: int,
    num_samples_y: int,
    randomness: Randomness,
    size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Indices of two training sets from two different corpora (cross-dataset setting)."""
    return randomness.subset(num_samples_x, size), randomness.subset(num_samples_y, size)


def sample_pairs(
    num_samples: int,
    num_pairs: int,
    randomness: Randomness,
) -> Tensor:
    """Indices of ``num_pairs`` known pairs. For one seed, smaller pair sets are subsets of larger ones."""
    return randomness.subset(num_samples, num_pairs)


def assert_disjoint(
    indices_x: Tensor,
    indices_y: Tensor,
) -> None:
    overlap = set(indices_x.tolist()) & set(indices_y.tolist())
    if overlap:
        raise AssertionError(f"The unpaired training sets share {len(overlap)} items.")


def midranks(
    scores: Tensor,
    true_scores: Tensor,
) -> Tensor:
    """Rank (1 = best) of the true key in each row. Tied keys share the average of their ranks."""
    num_higher = (scores > true_scores[:, None]).sum(dim=1).double()
    num_tied_others = (scores == true_scores[:, None]).sum(dim=1).double() - 1
    return num_higher + 0.5 * num_tied_others + 1


@torch.inference_mode()
def retrieval_metrics(
    aligner: Aligner,
    val_x: Tensor,
    val_y: Tensor,
    randomness: Randomness,
    block_size: int = 4096,
) -> dict[str, float]:
    """FOSCTTM and mean rank for row-aligned validation sets ``val_x`` and ``val_y``.

    FOSCTTM is the fraction of keys closer to a query than its true partner, averaged over queries (0 = perfect,
    0.5 = chance). Ties count half. Both sides are shuffled separately, and the scores are computed in blocks.

    Args:
        aligner: A fitted aligner.
        val_x: Validation queries ``[n, d_X]``.
        val_y: Validation keys ``[n, d_Y]``, where row ``i`` is the partner of ``val_x[i]``.
        randomness: Generators of the run, used for the shuffles.
        block_size: Number of queries scored at once.

    Returns:
        ``{"foscttm": ..., "mean_rank": ...}``.
    """
    num_samples = val_x.shape[0]
    order_x = randomness.permutation(num_samples)
    order_y = randomness.permutation(num_samples)
    shuffled_x, shuffled_y = val_x[order_x], val_y[order_y]
    partner = torch.argsort(order_y)[order_x]  # shuffled_y[partner[i]] is the partner of shuffled_x[i]
    ranks = torch.empty(num_samples, dtype=torch.float64)
    for start in range(0, num_samples, block_size):
        stop = min(start + block_size, num_samples)
        scores = aligner.similarity(shuffled_x[start:stop], shuffled_y).double().cpu()
        true_scores = scores[torch.arange(stop - start), partner[start:stop]]
        ranks[order_x[start:stop]] = midranks(scores, true_scores)  # in the original row order
    return {
        "foscttm": ((ranks - 1) / (num_samples - 1)).mean().item(),
        "mean_rank": ranks.mean().item(),
    }


@torch.inference_mode()
def zero_shot_accuracy(
    aligner: Aligner,
    images: Tensor,
    class_prompts: Tensor,
    labels: Tensor,
    randomness: Randomness,
) -> dict[str, float]:
    """Top-1 and top-5 accuracy of assigning each image to its most similar class prompt."""
    order_images = randomness.permutation(images.shape[0])
    order_classes = randomness.permutation(class_prompts.shape[0])
    scores = aligner.similarity(images[order_images], class_prompts[order_classes])
    predictions = order_classes[scores.topk(k=5, dim=1).indices.cpu()]  # back to class ids
    shuffled_labels = labels[order_images]
    return {
        "top1": (predictions[:, 0] == shuffled_labels).float().mean().item(),
        "top5": (predictions == shuffled_labels[:, None]).any(dim=1).float().mean().item(),
    }


@torch.inference_mode()
def label_transfer_accuracy(
    aligner: Aligner,
    val_x: Tensor,
    val_y: Tensor,
    labels_x: Tensor,
    labels_y: Tensor,
    randomness: Randomness,
    block_size: int = 4096,
) -> float:
    """Fraction of samples of X whose nearest neighbor in Y has the same label, such as the cell type."""
    order_y = randomness.permutation(val_y.shape[0])
    shuffled_y, shuffled_labels_y = val_y[order_y], labels_y[order_y]
    correct = 0
    for start in range(0, val_x.shape[0], block_size):
        nearest = aligner.similarity(val_x[start : start + block_size], shuffled_y).argmax(dim=1).cpu()
        correct += (shuffled_labels_y[nearest] == labels_x[start : start + block_size]).sum().item()
    return correct / val_x.shape[0]
