"""Shared geometry predicts unpaired alignment (Sec. 4.1): Fig. 3c and Fig. 14.

Computes four geometry scores on up to 10,000 paired items for each corpus and model pair. They are plotted against
the mean FOSCTTM of our aligner from ``experiments/unpaired.py``.

    python -m experiments.shared_geometry run
    python -m experiments.shared_geometry plot
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from experiments import plotting
from experiments.runner import Experiment, figures_dir, main, results_dir
from modalities import image_captions
from unpaired_rosetta.embeddings import embedding_path, load_rows, num_rows
from unpaired_rosetta.geometry import geometry_scores
from unpaired_rosetta.randomness import Randomness
from unpaired_rosetta.results import load_results

if TYPE_CHECKING:
    from torch import Tensor

SCORES = {"cka": "CKA", "mutual_knn": "mutual k-NN (k = 10)", "tsi": "TSI", "qsi": "QSI"}


@dataclass(frozen=True)
class GeometryRun:
    dataset: str  # a paired corpus
    model_x: str
    model_y: str
    modality_x: str = "vision"
    modality_y: str = "language"
    holdout: int = 0  # score only the last ``holdout`` items (NQ's validation set)
    num_samples: int = 10_000
    seed: int = 42


def configurations() -> list[GeometryRun]:
    return [
        GeometryRun(dataset, vision, language)
        for dataset in image_captions.TRAINING_DATASETS
        for vision in image_captions.VISION_MODELS
        for language in image_captions.LANGUAGE_MODELS
    ]


def paired_sample(
    config: GeometryRun,
    randomness: Randomness,
) -> tuple[Tensor, Tensor]:
    """``num_samples`` random paired items, each side centered."""
    path_x = embedding_path(config.dataset, config.modality_x, config.model_x)
    path_y = embedding_path(config.dataset, config.modality_y, config.model_y)
    num_items = config.holdout or num_rows(path_x)
    offset = num_rows(path_x) - config.holdout if config.holdout else 0
    indices = offset + randomness.subset(num_items, config.num_samples)
    samples_x, samples_y = load_rows(path_x, indices), load_rows(path_y, indices)
    return samples_x - samples_x.mean(dim=0), samples_y - samples_y.mean(dim=0)


def run(
    config: GeometryRun,
    num_workers: int = 1,
    num_threads: int = 1,
) -> dict:
    randomness = Randomness(config.seed)
    samples_x, samples_y = paired_sample(config, randomness)
    return geometry_scores(samples_x, samples_y, randomness.torch)


def plot() -> None:
    scores = load_results(results_dir("shared_geometry"))
    alignment = load_results(results_dir("unpaired"))
    alignment = alignment[(alignment["method"] == "ours") & alignment["dataset_y"].isna()]
    foscttm = alignment.groupby(["dataset", "model_x", "model_y"])["foscttm"].mean().reset_index()
    joined = foscttm.merge(scores, on=["dataset", "model_x", "model_y"])
    joined["corpus"] = pd.Categorical(
        joined["dataset"].map(image_captions.TRAINING_DATASETS), list(image_captions.TRAINING_DATASETS.values())
    )
    joined = joined.sort_values("corpus")
    clusters = joined["model_x"] + " / " + joined["model_y"]
    figure, axes = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
    statistics = []
    for axis, (score, label) in zip(axes, SCORES.items()):
        fit = plotting.linear_fit(joined[score], joined["foscttm"], clusters)
        plotting.fit_scatter(axis, joined, score, "foscttm", "corpus", fit)
        axis.set_xlabel(label)
        statistics.append(
            {
                "score": score,
                "r2": fit["r2"],
                "spearman": fit["spearman"],
                "slope": fit["slope"],
                "points": fit["num_points"],
            }
        )
    axes[0].set_ylabel("FOSCTTM of our unpaired alignment")
    axes[0].legend(fontsize=6, loc="lower left")
    plotting.save(figure, figures_dir("shared_geometry") / "fig3c_fig14_geometry", joined)
    pd.DataFrame(statistics).to_csv(figures_dir("shared_geometry") / "fig3c_fig14_fit_statistics.csv", index=False)
    print(pd.DataFrame(statistics).to_string(index=False))


EXPERIMENT = Experiment("shared_geometry", configurations, run, plot, memory_gb=64, hours=4)

if __name__ == "__main__":
    main(EXPERIMENT)
