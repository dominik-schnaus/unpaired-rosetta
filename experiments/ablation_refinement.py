"""Table 4 (Appendix A.2.1): FOSCTTM after no refinement, mini-vec2vec's refinement, or ours, from one read-out.

Each run initializes and reads out once with our method. Each refinement starts from the same randomness, so the rows
differ only in the refinement.

    python -m experiments.ablation_refinement run [--workers 10]
    python -m experiments.ablation_refinement plot
"""

import copy
from dataclasses import dataclass

from experiments.ablation_alignment import (
    SETTINGS,
    AblationRun,
    evaluate_variants,
    load_prepared_data,
    results_frame,
    stage_table,
)
from experiments.ablation_figures import save_table
from experiments.runner import Experiment, main, results_dir
from unpaired_rosetta.ablation.refinements import mini_vec2vec_refinement
from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.randomness import SEEDS
from unpaired_rosetta.results import load_results

NAME = "ablation_refinement"
ROWS = {"none": "no refinement", "mini-vec2vec": "mini-vec2vec", "ours": "ours"}


@dataclass(frozen=True)
class RefinementComparison:
    setting: str
    seed: int


def configurations_of_table() -> list[RefinementComparison]:
    return [RefinementComparison(setting, seed) for setting in SETTINGS for seed in SEEDS]


def run(
    config: RefinementComparison,
    num_workers: int = 1,
    num_threads: int = 1,
) -> dict:
    with torch_threads():  # fixed thread count for determinism, as in WassersteinProcrustes.fit
        aligner, samples_x, samples_y, randomness, run = load_prepared_data(
            AblationRun(config.setting, config.seed), num_workers, num_threads
        )
        correspondence, _ = aligner.initialize(samples_x, samples_y, randomness)
        readout = aligner.readout(samples_x, samples_y, correspondence, None, None, randomness)
        weights = {"none": readout}
        weights["mini-vec2vec"] = mini_vec2vec_refinement(samples_x, samples_y, readout, copy.deepcopy(randomness))
        weights["ours"] = aligner.refine(samples_x, samples_y, readout, copy.deepcopy(randomness))
        return evaluate_variants(aligner, run, weights)


def plot() -> None:
    frame = results_frame(load_results(results_dir(NAME)), "refinement", list(ROWS))
    table = stage_table(frame, ROWS, "refinement")
    save_table(table, NAME, "tab4_refinement", "Table 4: FOSCTTM after refining the same read-out")


EXPERIMENT = Experiment(NAME, configurations_of_table, run, plot, memory_gb=96, hours=24)

if __name__ == "__main__":
    main(EXPERIMENT)
