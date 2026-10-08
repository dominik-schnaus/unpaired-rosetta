"""Cluster matching as a low-rank Gromov-Wasserstein solver (Appendix A.2.1, Table 2).

Each run starts a rank-``r`` plan with one of five initializations and records its loss before and after 200 steps
of mirror descent. The table averages over seeds and ranks.

    python -m experiments.lowrank_gw run [--shard 0/8]
    python -m experiments.lowrank_gw plot
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.ablation_figures import COLORS, save_figure, save_table, set_style
from experiments.runner import Experiment, figures_dir, main, results_dir
from modalities import lowrank_benchmarks
from unpaired_rosetta.ablation import lowrank_gw
from unpaired_rosetta.randomness import SEEDS, Randomness
from unpaired_rosetta.results import load_results

if TYPE_CHECKING:
    from torch import Tensor

RANKS = [10, 20, 30, 50]
NUM_ITERATIONS = 200


@dataclass(frozen=True)
class LowRankRun:
    benchmark: str
    initialization: str
    rank: int
    seed: int
    num_samples: int = 1000


def configurations() -> list[LowRankRun]:
    return [
        LowRankRun(benchmark, initialization, rank, seed)
        for benchmark in lowrank_benchmarks.NAMES
        for initialization in lowrank_gw.INITIALIZATIONS
        for rank in RANKS
        for seed in SEEDS
    ]


def costs(
    benchmark: str,
    samples_x: Tensor,
    samples_y: Tensor,
) -> tuple[Tensor, Tensor]:
    if benchmark in ("snare", "splatter"):
        return lowrank_gw.geodesic_cost(samples_x), lowrank_gw.geodesic_cost(samples_y)
    return lowrank_gw.squared_euclidean_cost(samples_x), lowrank_gw.squared_euclidean_cost(samples_y)


def run(
    config: LowRankRun,
    num_workers: int = 1,
    num_threads: int = 1,
) -> dict:
    randomness = Randomness(config.seed)
    samples_x, samples_y, permutation = lowrank_benchmarks.load(config.benchmark, config.seed, config.num_samples)
    samples_x, samples_y = samples_x - samples_x.mean(0), samples_y - samples_y.mean(0)
    cost_x, cost_y = costs(config.benchmark, samples_x, samples_y)

    start = time.time()
    factors = lowrank_gw.initial_factors(samples_x, samples_y, config.rank, config.initialization, randomness)
    plan = lowrank_gw.dense_plan(factors)
    metrics = {
        "seconds_init": time.time() - start,
        "gw_loss_init": lowrank_gw.gw_loss(cost_x, cost_y, plan),
        "foscttm_init": lowrank_gw.plan_foscttm(plan, samples_y, permutation),
    }
    start = time.time()
    factors, history = lowrank_gw.mirror_descent(factors, cost_x, cost_y, NUM_ITERATIONS)
    plan = lowrank_gw.dense_plan(factors)
    metrics.update(
        seconds_mirror_descent=time.time() - start,
        gw_loss_final=lowrank_gw.gw_loss(cost_x, cost_y, plan),
        foscttm_final=lowrank_gw.plan_foscttm(plan, samples_y, permutation),
        mirror_descent_iterations=len(history) - 1,
    )
    return metrics


def plot() -> None:
    frame = load_results(results_dir(EXPERIMENT.name))
    set_style()
    means = frame.groupby(["initialization", "benchmark"])[["gw_loss_init", "gw_loss_final"]].mean()
    best = means["gw_loss_final"].groupby("benchmark").min()
    table = pd.DataFrame(index=pd.Index(lowrank_gw.INITIALIZATIONS, name="Initialization"))
    for benchmark, label in lowrank_benchmarks.NAMES.items():
        cells = []
        for initialization in lowrank_gw.INITIALIZATIONS:
            before, after = means.loc[(initialization, benchmark)]
            cells.append(f"{before:.4f} / {after:.4f}" + (" *" if after == best[benchmark] else ""))
        table[label.replace("$k$", "k")] = cells
    title = "Table 2: GW loss before / after mirror descent (mean over seeds and ranks, * lowest after)"
    save_table(table, EXPERIMENT.name, "tab2_lowrank_gw", title)

    figure, axes = plt.subplots(1, len(lowrank_benchmarks.NAMES), figsize=(3 * len(lowrank_benchmarks.NAMES), 2.8))
    for axis, (benchmark, label) in zip(axes, lowrank_benchmarks.NAMES.items()):
        sns.lineplot(
            frame[frame["benchmark"] == benchmark],
            x="rank",
            y="gw_loss_final",
            hue="initialization",
            hue_order=lowrank_gw.INITIALIZATIONS,
            palette=COLORS[: len(lowrank_gw.INITIALIZATIONS)],
            marker="o",
            errorbar="sd",
            ax=axis,
            legend=axis is axes[-1],
        )
        axis.set_title(label.replace("$k$", "k"))
        axis.set_xlabel("rank r")
        axis.set_ylabel("GW loss after mirror descent" if axis is axes[0] else "")
    axes[-1].legend(title="initialization", bbox_to_anchor=(1.02, 0.5), loc="center left")
    frame.groupby(["benchmark", "initialization", "rank"])[["gw_loss_init", "gw_loss_final"]].mean().to_csv(
        figures_dir(EXPERIMENT.name) / "tab2_loss_by_rank_values.csv"
    )
    save_figure(figure, EXPERIMENT.name, "tab2_loss_by_rank")


EXPERIMENT = Experiment("lowrank_gw", configurations, run, plot, memory_gb=16, hours=8)

if __name__ == "__main__":
    main(EXPERIMENT)
