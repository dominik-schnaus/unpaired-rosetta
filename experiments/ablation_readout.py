"""Table 3 (Appendix A.2.1): FOSCTTM of three read-outs of the same averaged correspondence, without refinement.

Each run computes our initialization once, which also gives the matched centers that relative representations
need. Each read-out starts from the same randomness, so the rows differ only in the read-out.

    python -m experiments.ablation_readout run [--workers 10]
    python -m experiments.ablation_readout plot
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
from unpaired_rosetta.ablation.readouts import direct_readout, relative_representation_readout
from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.randomness import SEEDS
from unpaired_rosetta.results import load_results

NAME = "ablation_readout"
ROWS = {
    "direct": "direct",
    "relative representations": "top-k relative representations",
    "ours": "batched Hungarian matching",
}


@dataclass(frozen=True)
class ReadoutComparison:
    setting: str
    seed: int


def configurations_of_table() -> list[ReadoutComparison]:
    return [ReadoutComparison(setting, seed) for setting in SETTINGS for seed in SEEDS]


def run(
    config: ReadoutComparison,
    num_workers: int = 1,
    num_threads: int = 1,
) -> dict:
    with torch_threads():  # fixed thread count for determinism, as in WassersteinProcrustes.fit
        ablation = AblationRun(config.setting, config.seed, readout="relative representations", refinement="none")
        aligner, samples_x, samples_y, randomness, run = load_prepared_data(ablation, num_workers, num_threads)
        correspondence, anchors = aligner.initialize(samples_x, samples_y, randomness)
        weights = {
            "direct": direct_readout(correspondence),
            "relative representations": relative_representation_readout(samples_x, samples_y, *anchors),
            "ours": aligner.readout(samples_x, samples_y, correspondence, None, None, copy.deepcopy(randomness)),
        }
        return evaluate_variants(aligner, run, weights)


def plot() -> None:
    frame = results_frame(load_results(results_dir(NAME)), "readout", list(ROWS))
    table = stage_table(frame, ROWS, "readout").rename_axis("Read-out")
    save_table(
        table,
        NAME,
        "tab3_readout",
        "Table 3: FOSCTTM after reading a map from the same averaged correspondence, without refinement",
    )


EXPERIMENT = Experiment(NAME, configurations_of_table, run, plot, memory_gb=96, hours=24)

if __name__ == "__main__":
    main(EXPERIMENT)
