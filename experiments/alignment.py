"""One alignment run: load two disjoint training sets, fit an aligner, and evaluate it on shuffled paired data.

Each alignment experiment of the paper is a list of ``AlignmentRun`` configurations passed to ``run_alignment``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from experiments.methods import build_aligner
from unpaired_rosetta.embeddings import embedding_path, load_rows, num_rows, storage_root
from unpaired_rosetta.evaluation import (
    disjoint_split,
    independent_split,
    label_transfer_accuracy,
    retrieval_metrics,
    sample_pairs,
    zero_shot_accuracy,
)
from unpaired_rosetta.randomness import Randomness

if TYPE_CHECKING:
    from torch import Tensor

    from unpaired_rosetta.evaluation import Aligner

# Folder of the class-prompt embeddings for zero-shot classification. CIFAR averages 18 CLIP templates per class.
# ImageNet-100 embeds "<class name>: <WordNet definition>" without a template.
CLASS_PROMPT_FOLDERS = {
    "CIFAR-10": "language_b953684b7ed3a62cbea9fe4e30935ee2",
    "CIFAR-100": "language_b953684b7ed3a62cbea9fe4e30935ee2",
    "ImageNet-100": "language",
}


@dataclass(frozen=True)
class AlignmentRun:
    dataset: str  # training corpus of X, and of Y unless ``dataset_y`` is set
    model_x: str
    model_y: str
    seed: int
    method: str = "ours"
    modality_x: str = "vision"
    modality_y: str = "language"
    dataset_y: str | None = None  # cross-dataset setting: Y comes from another corpus, so no pairs exist
    num_pairs: int = 0
    pairs_dataset: str | None = None  # corpus of the known pairs (default: the training corpus)
    validation: str = "coco_val2014"  # paired validation corpus (the training corpus if ``holdout`` > 0)
    holdout: int = 0  # the last ``holdout`` items of the training corpus are the validation set
    train_size: int | None = None  # items per side (default: half of the corpus)
    validation_size: int | None = None  # validation items (default: all)
    classification: tuple[str, ...] = ()
    labels: str | None = None  # label file of the validation corpus, for label transfer


def load_training_data(
    run: AlignmentRun,
    randomness: Randomness,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """The two unpaired training sets and the known pairs, all drawn with the run's randomness."""
    path_x = embedding_path(run.dataset, run.modality_x, run.model_x)
    path_y = embedding_path(run.dataset_y or run.dataset, run.modality_y, run.model_y)
    if run.dataset_y is None:
        indices_x, indices_y = disjoint_split(num_rows(path_x) - run.holdout, randomness, run.train_size)
    else:
        indices_x, indices_y = independent_split(num_rows(path_x), num_rows(path_y), randomness, run.train_size)
    samples_x, samples_y = load_rows(path_x, indices_x), load_rows(path_y, indices_y)
    if run.num_pairs == 0:
        return samples_x, samples_y, None, None
    pairs_dataset = run.pairs_dataset or run.dataset
    pair_path_x = embedding_path(pairs_dataset, run.modality_x, run.model_x)
    pair_path_y = embedding_path(pairs_dataset, run.modality_y, run.model_y)
    pairs = sample_pairs(num_rows(pair_path_x) - run.holdout, run.num_pairs, randomness)
    return samples_x, samples_y, load_rows(pair_path_x, pairs), load_rows(pair_path_y, pairs)


def load_validation_data(
    run: AlignmentRun,
    randomness: Randomness,
) -> tuple[Tensor, Tensor, Tensor]:
    """Paired validation rows in random order, and their indices. The evaluator shuffles each side again."""
    path_x = embedding_path(run.validation, run.modality_x, run.model_x)
    path_y = embedding_path(run.validation, run.modality_y, run.model_y)
    num_items = run.holdout or num_rows(path_x)
    offset = num_rows(path_x) - run.holdout if run.holdout else 0
    indices = offset + randomness.subset(num_items, run.validation_size)
    return load_rows(path_x, indices), load_rows(path_y, indices), indices


def evaluate(
    aligner: Aligner,
    run: AlignmentRun,
) -> dict:
    randomness = Randomness(run.seed)
    val_x, val_y, indices = load_validation_data(run, randomness)
    metrics = retrieval_metrics(aligner, val_x, val_y, randomness)
    metrics["num_validation_samples"] = val_x.shape[0]
    if run.labels is not None:
        labels = torch.load(storage_root() / "embeddings" / run.validation / run.labels)[indices]
        metrics["label_transfer_accuracy"] = label_transfer_accuracy(aligner, val_x, val_y, labels, labels, randomness)
    for dataset in run.classification:
        images = load_rows(embedding_path(dataset, run.modality_x, run.model_x))
        class_prompts = load_rows(embedding_path(dataset, CLASS_PROMPT_FOLDERS[dataset], run.model_y))
        labels = torch.load(storage_root() / "embeddings" / dataset / "labels.pt")
        accuracy = zero_shot_accuracy(aligner, images, class_prompts, labels, randomness)
        metrics[f"{dataset}_top1"] = accuracy["top1"]
        metrics[f"{dataset}_top5"] = accuracy["top5"]
    return metrics


def run_alignment(
    run: AlignmentRun,
    num_workers: int = 1,
    num_threads: int = 1,
) -> tuple[object, dict]:
    """Fit the run's method on its training data. Return the fitted aligner and its validation metrics."""
    start = time.time()
    randomness = Randomness(run.seed)
    samples_x, samples_y, paired_x, paired_y = load_training_data(run, randomness)
    aligner = build_aligner(run.method, num_workers=num_workers, num_threads=num_threads)
    aligner.fit(samples_x, samples_y, paired_x, paired_y, randomness)
    metrics = evaluate(aligner, run)
    metrics["fit_and_evaluation_seconds"] = time.time() - start
    return aligner, metrics
