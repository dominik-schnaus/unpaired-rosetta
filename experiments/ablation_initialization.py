"""Table 1 (Appendix A.2.1): FOSCTTM of each initialization with our read-out and no refinement.

python -m experiments.ablation_initialization run [--workers 10]
python -m experiments.ablation_initialization plot
"""

from experiments.ablation_alignment import AblationRun, configurations, run_ablation, stage_table
from experiments.ablation_figures import save_table
from experiments.runner import Experiment, main, results_dir
from unpaired_rosetta.results import load_results

NAME = "ablation_initialization"
ROWS = {
    "PCA": "PCA heuristic",
    "sorted": "sorted heuristic",
    "GW (random)": "Gromov-Wasserstein (random)",
    "GW (uniform)": "Gromov-Wasserstein (uniform)",
    "2-opt, C=20": "mini-vec2vec (k-means + 2-opt, C = 20)",
    "2-opt, C=30": "mini-vec2vec (k-means + 2-opt, C = 30)",
    "ours": "ours (k-means + MPOpt)",
}


def configurations_of_table() -> list[AblationRun]:
    runs = configurations(initialization=["PCA", "sorted", "GW (random)", "GW (uniform)", "ours"], refinement=["none"])
    return runs + configurations(initialization=["2-opt"], num_clusters=[20, 30], refinement=["none"])


def row_name(
    initialization: str,
    num_clusters: int,
) -> str:
    return f"2-opt, C={num_clusters}" if initialization == "2-opt" else initialization


def plot() -> None:
    results = load_results(results_dir(NAME))
    results["variant"] = [row_name(*values) for values in zip(results["initialization"], results["num_clusters"])]
    table = stage_table(results, ROWS, "variant").rename_axis("Initialization")
    save_table(
        table,
        NAME,
        "tab1_initialization",
        "Table 1: FOSCTTM of every initialization with the same read-out and no refinement",
    )


EXPERIMENT = Experiment(NAME, configurations_of_table, run_ablation, plot, memory_gb=96, hours=24)

if __name__ == "__main__":
    main(EXPERIMENT)
