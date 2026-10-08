"""Figure 7 (Appendix A.2.1): MPOpt, the QAP solver of our initialization, on the class-matching benchmark of Schnaus
et al. (2025).

Runs ``MPOptQAPSolver`` on 10 to 100 CIFAR-100 classes with the settings of the paper and with the "equal time"
budget, and reports its cost, lower bound and run time. All other solvers of Figure 7 (LocalCKA, entropic OT, FAQ,
MPOpt, Gurobi, Hahn-Grant) are the published runs of Schnaus et al. (2025). Reproduce them with the official code,
github.com/dominik-schnaus/itsamatch, ``itsamatch/experiments/solver_comparison_larger.py``.

    python -m experiments.qap_solvers run
    python -m experiments.qap_solvers plot
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from matplotlib import pyplot as plt

from experiments.ablation_figures import COLORS, TEXT, save_figure, save_table, set_style
from experiments.runner import Experiment, main, results_dir
from modalities import qap_benchmark
from unpaired_rosetta.ablation.qap_solvers import MPOptBudget, QAPResult, solve_with_bound
from unpaired_rosetta.results import load_results

NAME = "qap_solvers"
SOLVER = "mpopt_paper"
# "Equal time" budgets: the settings of the paper with early stopping off and this many batches, so that MPOpt runs
# about as long as Hahn-Grant on each size (1, 15, 84 and 389 seconds up to 40 classes, and its limit of 5400 seconds
# beyond). The number is the Hahn-Grant time over the seconds per batch of the default runs.
EQUAL_TIME_BATCHES = {
    10: 287,
    20: 268,
    30: 273,
    40: 383,
    50: 1373,
    60: 624,
    70: 372,
    80: 189,
    90: 131,
    100: 86,
}


@dataclass(frozen=True)
class QAPRun:
    solver: str
    size: int
    budget: str = "default"  # or "equal_time"
    seed: int = qap_benchmark.SEED


def budget(
    config: QAPRun,
) -> MPOptBudget:
    if config.budget == "default":
        return MPOptBudget()
    batches = EQUAL_TIME_BATCHES[config.size]
    # Early stopping is off (k is the number of batches), so the run time depends only on the budget.
    return MPOptBudget(max_batches=batches, stopping_k=batches)


def configurations() -> list[QAPRun]:
    return [QAPRun(SOLVER, size, budget) for budget in ("default", "equal_time") for size in qap_benchmark.SIZES]


def solve(
    config: QAPRun,
) -> QAPResult:
    problem = qap_benchmark.gromov_wasserstein_problem(config.size)
    return solve_with_bound(problem.cost1, problem.cost2, problem.constant.item(), budget(config), config.seed)


def run(
    config: QAPRun,
    num_workers: int = 1,
    num_threads: int = 1,
) -> dict:
    torch.set_num_threads(4)
    result = solve(config)
    return {
        "gw_cost": qap_benchmark.gromov_wasserstein_cost(config.size, result.permutation),
        "bound": result.bound,
        "seconds": result.seconds,
        "batches": result.batches,
        "matching_accuracy": (result.permutation == torch.arange(config.size)).double().mean().item(),
        "permutation": result.permutation.tolist(),
    }


def normalized_results() -> pd.DataFrame:
    results = load_results(results_dir(NAME))
    results = results[results["solver"] == SOLVER].copy()
    results["cost"] = results["gw_cost"] / results["size"] ** 2
    results["normalized_bound"] = pd.to_numeric(results["bound"], errors="coerce") / results["size"] ** 2
    return results


def plot() -> None:
    results = normalized_results()
    set_style()
    figure, axes = plt.subplots(figsize=(6.5, 4))
    for color, budget in zip(COLORS, ("default", "equal_time")):
        rows = results[results["budget"] == budget].sort_values("size")
        label = "MPOpt + heuristic" + (" (equal time)" if budget == "equal_time" else "")
        axes.plot(rows["size"], rows["cost"], color=color, marker="o", markersize=4, linewidth=1.5, label=label)
        axes.plot(rows["size"], rows["normalized_bound"], color=color, linestyle="--", linewidth=1.5)
    axes.plot([], [], color=TEXT, label="cost")
    axes.plot([], [], color=TEXT, linestyle="--", label="bound")
    axes.set(xlabel="problem size C", ylabel="GW cost / bound", xticks=qap_benchmark.SIZES)
    axes.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    axes.set_title("Figure 7a: MPOpt + heuristic, cost (solid) and bound (dashed), normalized by C²", fontsize=10)
    save_figure(figure, NAME, "fig7a_cost_and_bound")
    save_table(
        results.pivot_table(index="size", columns="budget", values=["cost", "normalized_bound"]),
        NAME,
        "fig7a_values",
        "Figure 7a: normalized cost and bound of MPOpt + heuristic",
    )
    time_table(results)


def time_table(
    results: pd.DataFrame,
) -> None:
    """Figure 7b: the MPOpt + heuristic columns (cost, time, cost with the equal-time budget)."""

    def column(
        budget: str,
        value: str,
    ) -> pd.Series:
        return results[results["budget"] == budget].set_index("size")[value]

    table = pd.DataFrame(
        {
            "cost": column("default", "cost").map("{:.3f}".format),
            "time [s]": column("default", "seconds").map("{:.0f}".format),
            "equal time": column("equal_time", "cost").map("{:.3f}".format),
            "equal time [s]": column("equal_time", "seconds").map("{:.0f}".format),
        }
    ).rename_axis("Size")
    save_table(table, NAME, "fig7b_cost_and_time", "Figure 7b: MPOpt + heuristic columns")


EXPERIMENT = Experiment(NAME, configurations, run, plot, memory_gb=64, hours=8)

if __name__ == "__main__":
    main(EXPERIMENT)
