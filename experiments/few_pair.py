"""Few-pair alignment (Sec. 4.4): Fig. 5.

Aligns DINOv2 and Qwen3-8B on MS COCO with 0 to 1,000 known image-caption pairs. Every method gets the same pairs.
Our zero-pair point is taken from ``experiments/unpaired.py``, which runs the same configuration.

    python -m experiments.few_pair run
    python -m experiments.few_pair plot
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import plotting
from experiments.alignment import AlignmentRun, run_alignment
from experiments.runner import Experiment, figures_dir, main, results_dir
from experiments.unpaired import CLASSIFICATION
from modalities.image_captions import QWEN3_GEN
from unpaired_rosetta.randomness import SEEDS
from unpaired_rosetta.results import load_results

VISION_MODEL = "dinov2_vit-b14@224_mean"
NUM_PAIRS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
BASELINES = ["linear", "orthogonal", "ASIF", "LocalCKA", "SUE", "STRUCTURE", "SOTAlign"]
METHODS = BASELINES + ["ours"]
PANELS = {
    "foscttm": "FOSCTTM ↓",
    "CIFAR-10_top1": "CIFAR-10 Top-1 ↑",
    "CIFAR-100_top5": "CIFAR-100 Top-5 ↑",
    "ImageNet-100_top5": "ImageNet-100 Top-5 ↑",
}


def configurations() -> list[AlignmentRun]:
    return [
        AlignmentRun(
            "coco_train2014", VISION_MODEL, QWEN3_GEN, seed, method, num_pairs=num_pairs, classification=CLASSIFICATION
        )
        for method in METHODS
        for num_pairs in NUM_PAIRS
        for seed in SEEDS
        if not (method == "ours" and num_pairs == 0)
    ]


def run(
    config: AlignmentRun,
    num_workers: int,
    num_threads: int,
) -> dict:
    _, metrics = run_alignment(config, num_workers, num_threads)
    return metrics


def plot() -> None:
    frame = load_results(results_dir("few_pair"))
    unpaired = load_results(results_dir("unpaired"))
    ours_without_pairs = unpaired[
        (unpaired["method"] == "ours")
        & (unpaired["dataset"] == "coco_train2014")
        & unpaired["dataset_y"].isna()
        & (unpaired["model_x"] == VISION_MODEL)
        & (unpaired["model_y"] == QWEN3_GEN)
    ]
    frame = pd.concat([frame, ours_without_pairs])
    figure, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True)
    numbers = []
    colors = dict(zip(BASELINES, ["#E37222", "#A2AD00", "#9C9A8E", "#69085A", "#C4071B", "#F7B11E", "#2E8540"]))
    colors["ours"] = plotting.OURS_COLOR
    for axis, (metric, title) in zip(axes.flat, PANELS.items()):
        percent = 1 if metric == "foscttm" else 100
        for method in METHODS:
            summary = plotting.mean_and_std(frame[frame["method"] == method], ["num_pairs"], metric).sort_values(
                "num_pairs"
            )
            x = summary["num_pairs"].replace(0, 0.5)  # place 0 at the left end of the log axis
            axis.plot(
                x,
                percent * summary["mean"],
                marker="o",
                markersize=3,
                color=colors[method],
                label=method,
                linewidth=2 if method == "ours" else 1,
            )
            axis.fill_between(
                x,
                percent * (summary["mean"] - summary["std"]),
                percent * (summary["mean"] + summary["std"]),
                color=colors[method],
                alpha=0.15,
                linewidth=0,
            )
            numbers.append(summary.assign(method=method, metric=metric))
        axis.set_xscale("symlog", linthresh=1)
        axis.set_xticks([0.5, 1, 10, 100, 1000], ["0", "1", "10", "100", "1000"])
        axis.set_title(title, fontsize=9)
    axes[1, 0].set_xlabel("number of known pairs")
    axes[1, 1].set_xlabel("number of known pairs")
    axes[0, 0].legend(fontsize=6, ncol=2)
    plotting.save(figure, figures_dir("few_pair") / "fig5_few_pair", pd.concat(numbers))
    text_numbers(pd.concat(numbers))


def text_numbers(
    numbers: pd.DataFrame,
) -> None:
    """The ratios quoted in Sec. 4.4: our FOSCTTM against the strongest baseline for each number of pairs."""
    foscttm = numbers[numbers["metric"] == "foscttm"].pivot(index="num_pairs", columns="method", values="mean")
    strongest = foscttm[BASELINES].min(axis=1)
    lines = [
        f"{pairs} pairs: ours {foscttm.loc[pairs, 'ours']:.4f}, strongest baseline {strongest.loc[pairs]:.4f}, "
        f"ratio {strongest.loc[pairs] / foscttm.loc[pairs, 'ours']:.1f}x"
        for pairs in foscttm.index
        if not np.isnan(foscttm.loc[pairs, "ours"])
    ]
    path = figures_dir("few_pair") / "text_numbers.md"
    path.write_text("\n".join(f"- {line}" for line in lines) + "\n")


EXPERIMENT = Experiment("few_pair", configurations, run, plot, memory_gb=96, hours=12)

if __name__ == "__main__":
    main(EXPERIMENT)
