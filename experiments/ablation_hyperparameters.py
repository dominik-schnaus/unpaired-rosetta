"""Figure 8 (Appendix A.2.2): FOSCTTM as each of four hyperparameters varies, with the others fixed.

Clusters ``C`` and batch size ``b`` get one run per value without refinement. For restarts ``S``, one run computes
100 restarts and reads out the average of the first ``S``. Restart ``s`` draws the same random numbers whatever the
total, so the result equals a run with ``S`` restarts. For refinement iterations ``R``, one run refines to 100
iterations and evaluates at each grid value. The evaluations share the random stream with the refinement, so they
change the batches of later iterations.

    python -m experiments.ablation_hyperparameters run [--workers 10]
    python -m experiments.ablation_hyperparameters plot
"""

from dataclasses import dataclass

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from experiments.ablation_alignment import (
    SETTINGS,
    AblationRun,
    load_prepared_data,
    run_ablation,
    validation_after_split,
)
from experiments.ablation_figures import COLORS, DATASET_LABELS, save_figure, save_table, set_style
from experiments.runner import Experiment, main, results_dir
from unpaired_rosetta.ablation.initializations import ClusterMatching, average_cross_covariance
from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.evaluation import retrieval_metrics
from unpaired_rosetta.randomness import SEEDS, Randomness
from unpaired_rosetta.results import load_results

NAME = "ablation_hyperparameters"
GRIDS = {
    "C": [5, 10, 15, 20, 25, 30, 35, 40, 50],
    "b": [125, 250, 500, 1000, 2500, 5000, 10_000, 20_000],
    "S": [1, 2, 3, 5, 10, 20, 30, 50, 100],
    "R": [0, 1, 2, 5, 10, 25, 50, 100],
}
AXES = {  # x label and whether the x axis is logarithmic
    "C": ("number of clusters C", False),
    "b": ("batch size b", True),
    "S": ("initialization restarts S", True),
    "R": ("refinement iterations R", False),
}


@dataclass(frozen=True)
class HyperparameterRun:
    setting: str
    seed: int
    parameter: str  # "C", "b", "S" or "R"
    value: int | None = None  # None for S and R, where one run covers the whole grid


def configurations_of_figure() -> list[HyperparameterRun]:
    runs = []
    for setting in SETTINGS:
        for seed in SEEDS:
            runs += [
                HyperparameterRun(setting, seed, parameter, value) for parameter in "Cb" for value in GRIDS[parameter]
            ]
            runs += [HyperparameterRun(setting, seed, parameter) for parameter in "SR"]
    return runs


def run(
    config: HyperparameterRun,
    num_workers: int = 1,
    num_threads: int = 1,
) -> dict:
    if config.parameter == "C":
        return run_ablation(
            AblationRun(config.setting, config.seed, num_clusters=config.value, refinement="none"),
            num_workers,
            num_threads,
        )
    if config.parameter == "b":
        return run_ablation(
            AblationRun(config.setting, config.seed, batch_size=config.value, refinement="none"),
            num_workers,
            num_threads,
        )
    if config.parameter == "S":
        return restart_sweep(config, num_workers, num_threads)
    if config.parameter == "R":
        return iteration_sweep(config, num_workers, num_threads)
    raise ValueError(f"Unknown parameter: {config.parameter}")


def restart_sweep(
    config: HyperparameterRun,
    num_workers: int,
    num_threads: int,
) -> dict:
    with torch_threads():  # fixed thread count for determinism, as in WassersteinProcrustes.fit
        aligner, samples_x, samples_y, randomness, run = load_prepared_data(
            AblationRun(config.setting, config.seed), num_workers, num_threads
        )
        val_x, val_y = validation_after_split(run, randomness)
        matched_centers = ClusterMatching(num_restarts=max(GRIDS["S"]), num_workers=num_workers)(
            samples_x, samples_y, randomness
        )
        metrics = {}
        for num_restarts in GRIDS["S"]:
            correspondence = average_cross_covariance(matched_centers[:num_restarts])
            readout_randomness = Randomness(config.seed)  # same read-out draw for every S
            aligner.weight = aligner.readout(samples_x, samples_y, correspondence, None, None, readout_randomness)
            metrics.update(
                {
                    f"S={num_restarts}/{key}": value
                    for key, value in retrieval_metrics(aligner, val_x, val_y, readout_randomness).items()
                }
            )
        return metrics


def iteration_sweep(
    config: HyperparameterRun,
    num_workers: int,
    num_threads: int,
) -> dict:
    with torch_threads():  # fixed thread count for determinism, as in WassersteinProcrustes.fit
        aligner, samples_x, samples_y, randomness, run = load_prepared_data(
            AblationRun(config.setting, config.seed), num_workers, num_threads
        )
        val_x, val_y = validation_after_split(run, randomness)
        correspondence, _ = aligner.initialize(samples_x, samples_y, randomness)
        stream = Randomness(config.seed)  # shared by read-out, refinement and evaluations
        weight = aligner.readout(samples_x, samples_y, correspondence, None, None, stream)
        metrics, done = {}, 0
        for num_iterations in GRIDS["R"]:
            for _ in range(num_iterations - done):
                weight = aligner.refinement_step(samples_x, samples_y, weight, None, None, stream)
            done = num_iterations
            aligner.weight = weight
            metrics.update(
                {
                    f"R={num_iterations}/{key}": value
                    for key, value in retrieval_metrics(aligner, val_x, val_y, stream).items()
                }
            )
        return metrics


def long_frame(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """One row per setting, seed, parameter and value, with its FOSCTTM."""
    rows = []
    for _, result in results.iterrows():
        if result["parameter"] in "Cb":
            rows.append((result["setting"], result["seed"], result["parameter"], result["value"], result["foscttm"]))
            continue
        for value in GRIDS[result["parameter"]]:
            column = f"{result['parameter']}={value}/foscttm"
            if column in result and pd.notna(result[column]):
                rows.append((result["setting"], result["seed"], result["parameter"], value, result[column]))
    return pd.DataFrame(rows, columns=["setting", "seed", "parameter", "value", "foscttm"])


def plot() -> None:
    frame = long_frame(load_results(results_dir(NAME)))
    frame["benchmark"] = frame["setting"].map(DATASET_LABELS)
    order = [DATASET_LABELS[setting] for setting in ("coco_train2014", "nq", "SNARE-seq")]
    set_style()
    figure, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for panel, (parameter, (label, logarithmic)) in zip(axes.flat, AXES.items()):
        data = frame[frame["parameter"] == parameter]
        sns.lineplot(
            data=data,
            x="value",
            y="foscttm",
            hue="benchmark",
            hue_order=order,
            palette=COLORS[: len(order)],
            errorbar="sd",
            marker="o",
            markersize=5,
            linewidth=2,
            ax=panel,
            legend=parameter == "C",
        )
        if logarithmic:
            panel.set_xscale("log")
            panel.set_xticks(GRIDS[parameter][::2], labels=[str(value) for value in GRIDS[parameter][::2]])
            panel.minorticks_off()
        panel.set_xlabel(label)
        panel.set_ylabel("FOSCTTM")
        panel.set_ylim(bottom=0)
    figure.suptitle("Figure 8: FOSCTTM against each hyperparameter (mean ± std over seeds)")
    figure.tight_layout()
    save_figure(figure, NAME, "fig8_hyperparameters")
    summary = frame.groupby(["parameter", "value", "benchmark"])["foscttm"].agg(["mean", "std", "count"]).reset_index()
    summary["foscttm"] = [
        f"{mean:.4f} ± {std:.4f} ({count})"
        for mean, std, count in zip(summary["mean"], summary["std"], summary["count"])
    ]
    table = summary.pivot(index=["parameter", "value"], columns="benchmark", values="foscttm")
    save_table(table, NAME, "fig8_hyperparameters_values", "Figure 8: FOSCTTM (mean ± std, number of seeds)")


EXPERIMENT = Experiment(NAME, configurations_of_figure, run, plot, memory_gb=96, hours=24)

if __name__ == "__main__":
    main(EXPERIMENT)
