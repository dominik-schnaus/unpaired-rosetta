"""Unpaired vision-language alignment (Sec. 4.1).

Produces Fig. 3a (FOSCTTM on MS COCO, with the cross-dataset column of ``experiments/cross_dataset.py``), Fig. 3b
(zero-shot accuracy), and Figs. 9-12 (FOSCTTM and zero-shot accuracy on MS COCO, SPC, DCI and DOCCI).

Aligns 7 vision models with 3 language models on 4 corpora using our aligner, mini-vec2vec and vec2vec. Each run is
evaluated on MS COCO 2014 val and by zero-shot classification.

    python -m experiments.unpaired run [--slurm --jobs 50 --workers 4 --threads 4]
    python -m experiments.unpaired plot
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from experiments import plotting
from experiments.alignment import AlignmentRun, run_alignment
from experiments.runner import Experiment, figures_dir, main, results_dir
from modalities import image_captions
from unpaired_rosetta.randomness import SEEDS
from unpaired_rosetta.results import load_results

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes

METHODS = ["ours", "mini-vec2vec", "vec2vec"]
CLASSIFICATION = tuple(image_captions.CLASSIFICATION_DATASETS)


def configurations() -> list[AlignmentRun]:
    return [
        AlignmentRun(dataset, vision, language, seed, method, classification=CLASSIFICATION)
        for method in METHODS
        for dataset in image_captions.TRAINING_DATASETS
        for vision in image_captions.VISION_MODELS
        for language in image_captions.LANGUAGE_MODELS
        for seed in SEEDS
    ]


def run(
    config: AlignmentRun,
    num_workers: int,
    num_threads: int,
) -> dict:
    _, metrics = run_alignment(config, num_workers, num_threads)
    return metrics


def readable(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add display names of the models, in the order of the figures."""
    frame = frame.copy()
    frame["vision"] = pd.Categorical(
        frame["model_x"].map(image_captions.VISION_MODELS), list(image_captions.VISION_MODELS.values())
    )
    frame["language"] = pd.Categorical(
        frame["model_y"].map(image_captions.LANGUAGE_MODELS), list(image_captions.LANGUAGE_MODELS.values())
    )
    return frame


def foscttm_panels(
    axes: Iterable[Axes],
    frame: pd.DataFrame,
    columns: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """One heatmap per language model, with methods as columns and vision models as rows."""
    numbers = []
    for axis, language in zip(axes, image_captions.LANGUAGE_MODELS.values()):
        means, stds = pd.DataFrame(), pd.DataFrame()
        for column_name, results in columns:
            summary = plotting.mean_and_std(results[results["language"] == language], ["vision"], "foscttm")
            summary = summary.set_index("vision").reindex(image_captions.VISION_MODELS.values())
            means[column_name], stds[column_name] = summary["mean"], summary["std"]
            numbers.append(summary.reset_index().assign(language=language, column=column_name))
        plotting.heatmap(axis, means, stds, plotting.FOSCTTM_COLORS, 0.0, 0.5)
        axis.set_title(language, fontsize=8)
        axis.tick_params(labelsize=6)
    return pd.concat(numbers)


def accuracy_panels(
    axes: Iterable[Axes],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """One heatmap per classification benchmark, with vision models as rows and language models as columns."""
    numbers = []
    for axis, (dataset, k) in zip(axes, image_captions.CLASSIFICATION_DATASETS.items()):
        metric = f"{dataset}_{k}"
        summary = plotting.mean_and_std(frame, ["vision", "language"], metric)
        means = summary.pivot(index="vision", columns="language", values="mean")
        stds = summary.pivot(index="vision", columns="language", values="std")
        chance = 1 / 10 if k == "top1" else 5 / 100
        plotting.heatmap(axis, means, stds, plotting.ACCURACY_COLORS, chance, 1.0, percent=True)
        axis.set_title(f"{dataset} {k.replace('top', 'Top-')}", fontsize=8)
        axis.tick_params(labelsize=6)
        numbers.append(summary.assign(metric=metric))
    return pd.concat(numbers)


def dataset_figure(
    frame: pd.DataFrame,
    dataset: str,
    name: str,
    extra_column: tuple[str, pd.DataFrame] | None = None,
) -> None:
    """Figs. 3a and 9-12: FOSCTTM of each method (top) and zero-shot accuracy of ours (bottom) on one corpus."""
    results = frame[(frame["dataset"] == dataset) & frame["dataset_y"].isna()]
    columns = [(method, results[results["method"] == method]) for method in ["ours", "mini-vec2vec", "vec2vec"]]
    if extra_column is not None:
        columns.insert(0, extra_column)
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.5), gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    foscttm_numbers = foscttm_panels(axes[0], results, columns)
    accuracy_numbers = accuracy_panels(axes[1], results[results["method"] == "ours"])
    for axis in list(axes[0][1:]) + list(axes[1][1:]):
        axis.set_yticklabels([])
    figure.suptitle(f"Unpaired alignment on {image_captions.TRAINING_DATASETS[dataset]}", fontsize=10)
    plotting.save(figure, figures_dir("unpaired") / name, pd.concat([foscttm_numbers, accuracy_numbers]))


def zero_shot_ranges(
    frame: pd.DataFrame,
) -> None:
    """Fig. 3b: for each vision model, the mean accuracy over the language models (dot) and their range (line)."""
    ours = frame[(frame["method"] == "ours") & (frame["dataset"] == "coco_train2014") & frame["dataset_y"].isna()]
    figure, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharey=True)
    numbers = []
    for axis, (dataset, k) in zip(axes, image_captions.CLASSIFICATION_DATASETS.items()):
        per_pair = ours.groupby(["vision", "language"], observed=True)[f"{dataset}_{k}"].mean().reset_index()
        per_vision = per_pair.groupby("vision", observed=True)[f"{dataset}_{k}"].agg(["mean", "min", "max"]) * 100
        rows = list(range(len(per_vision)))
        axis.hlines(rows, per_vision["min"], per_vision["max"], color=plotting.OURS_COLOR, linewidth=1.5)
        axis.plot(per_vision["mean"], rows, "o", color=plotting.OURS_COLOR, markersize=4)
        axis.axvline(10 if k == "top1" else 5, color="gray", linestyle="--", linewidth=0.8, label="chance")
        axis.set_yticks(rows, per_vision.index)
        axis.invert_yaxis()
        axis.set_title(f"{dataset}\n{k.replace('top', 'Top-')}", fontsize=8)
        axis.set_xlim(left=0)
        numbers.append(per_vision.reset_index().assign(metric=f"{dataset}_{k}"))
    axes[1].set_xlabel("zero-shot accuracy (%)")
    plotting.save(figure, figures_dir("unpaired") / "fig3b_zero_shot", pd.concat(numbers))


def plot() -> None:
    frame = readable(load_results(results_dir("unpaired")))
    crossed = readable(load_results(results_dir("cross_dataset")))
    crossed_column = ("ours†", crossed[crossed["method"] == "ours"]) if len(crossed) else None
    dataset_figure(frame, "coco_train2014", "fig3a_fig9_coco", crossed_column)
    for figure_number, dataset in [
        (10, "StanfordParagraphCaptioning"),
        (11, "DenselyCaptionedImages"),
        (12, "DOCCIDataset"),
    ]:
        dataset_figure(frame, dataset, f"fig{figure_number}_{dataset}")
    zero_shot_ranges(frame)
    summary_numbers(frame, crossed)


def summary_numbers(
    frame: pd.DataFrame,
    crossed: pd.DataFrame,
) -> None:
    """The averages quoted in the text of Sec. 4.1."""
    ours_coco = frame[(frame["method"] == "ours") & (frame["dataset"] == "coco_train2014")]
    lines = [f"average FOSCTTM of ours on MS COCO over 21 model pairs: {ours_coco['foscttm'].mean():.3f}"]
    for dataset, k in image_captions.CLASSIFICATION_DATASETS.items():
        lines.append(f"average {dataset} {k} accuracy on MS COCO: {100 * ours_coco[f'{dataset}_{k}'].mean():.1f}%")
    schnaus_pair = ours_coco[(ours_coco["model_x"] == "dinov2_vit-g14@224_mean") & (ours_coco["model_y"] == "mpnet")]
    lines.append(f"CIFAR-10 top-1 of DINOv2 G/14 x MPNet: {100 * schnaus_pair['CIFAR-10_top1'].mean():.1f}%")
    if len(crossed):
        lines.append(
            f"average FOSCTTM of ours, MS COCO images x SPC captions: {crossed[crossed['method'] == 'ours']['foscttm'].mean():.3f}"
        )
    path = figures_dir("unpaired") / "text_numbers.md"
    path.write_text("\n".join(f"- {line}" for line in lines) + "\n")
    print(path.read_text())


EXPERIMENT = Experiment("unpaired", configurations, run, plot, memory_gb=96, hours=12)

if __name__ == "__main__":
    main(EXPERIMENT)
