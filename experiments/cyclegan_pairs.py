"""Some domains need no alignment (App. C.4): Table 11.

Five modality pairs from the CycleGAN literature embed both sides with one encoder, so the identity map is a
baseline. We compare the identity map, the identity map after centering and normalizing, and our aligner, and report
FOSCTTM on the first 1,024 validation items, the slice an earlier study used.

Writes ``tab11_cyclegan_pairs.{png,csv,md}`` to ``$UNPAIRED_ROSETTA_ROOT/figures/cyclegan_pairs/``.

    python -m experiments.cyclegan_pairs run
    python -m experiments.cyclegan_pairs plot
"""

import pandas as pd

from experiments import plotting, shared_geometry
from experiments.alignment import AlignmentRun, load_validation_data, run_alignment
from experiments.other_domains import markdown
from experiments.runner import Experiment, figures_dir, main, results_dir
from experiments.shared_geometry import GeometryRun
from modalities import cyclegan_pairs
from unpaired_rosetta.evaluation import retrieval_metrics
from unpaired_rosetta.randomness import SEEDS, Randomness
from unpaired_rosetta.results import load_results

METHODS = {"identity": "identity", "identity (centred+norm)": "centred+norm", "ours": "ours"}  # name -> column
SLICE = 1024


def configurations() -> list[AlignmentRun]:
    return [
        AlignmentRun(
            x.dataset,
            x.model,
            y.model,
            seed,
            method,
            modality_x=x.modality,
            modality_y=y.modality,
            validation=x.dataset,
        )
        for x, y in cyclegan_pairs.PAIRS.values()
        for method in METHODS
        for seed in SEEDS
    ]


def run(
    config: AlignmentRun,
    num_workers: int,
    num_threads: int,
) -> dict:
    aligner, metrics = run_alignment(config, num_workers, num_threads)
    randomness = Randomness(config.seed)
    val_x, val_y, _ = load_validation_data(config, randomness)  # same order as in the evaluation
    metrics[f"foscttm_{SLICE}"] = retrieval_metrics(aligner, val_x[:SLICE], val_y[:SLICE], randomness)["foscttm"]
    if config.method == "ours":
        scored = GeometryRun(
            config.validation, config.model_x, config.model_y, config.modality_x, config.modality_y, seed=config.seed
        )
        metrics |= shared_geometry.run(scored)
    return metrics


def plot() -> None:
    frame = load_results(results_dir("cyclegan_pairs"))
    settings = {(x.dataset, x.modality, y.modality): name for name, (x, y) in cyclegan_pairs.PAIRS.items()}
    frame["setting"] = [settings[key] for key in zip(frame["dataset"], frame["modality_x"], frame["modality_y"])]
    foscttm = frame.pivot_table(index="setting", columns="method", values=f"foscttm_{SLICE}", aggfunc="mean")
    geometry = frame[frame["method"] == "ours"].groupby("setting")[["cka", "mutual_knn"]].mean()
    order = list(cyclegan_pairs.PAIRS)
    numbers = foscttm[list(METHODS)].join(geometry).reindex(order)
    table = pd.DataFrame({"Setting": order, "Encoder (both sides)": [cyclegan_pairs.ENCODERS[name] for name in order]})
    for method, column in METHODS.items():
        table[f"{column} FOSCTTM ↓"] = [f"{value:.3f}" for value in numbers[method]]
    table["CKA ↑"] = [f"{value:.3f}" for value in numbers["cka"]]
    table["mutual k-NN ↑"] = [f"{value:.3f}" for value in numbers["mutual_knn"]]
    # Bold: the lowest FOSCTTM in each row.
    best = numbers[list(METHODS)].eq(numbers[list(METHODS)].min(axis=1), axis=0).to_numpy()
    bold = pd.DataFrame(False, index=table.index, columns=table.columns)
    bold.iloc[:, 2:5] = best
    output = figures_dir("cyclegan_pairs") / "tab11_cyclegan_pairs"
    figure = plotting.table_figure(table, "Table 11: modality pairs from the CycleGAN literature", bold)
    plotting.save(figure, output, numbers.reset_index(names="setting"))
    output.with_suffix(".md").write_text(markdown(table))
    print(table.to_string(index=False))


EXPERIMENT = Experiment("cyclegan_pairs", configurations, run, plot, memory_gb=32, hours=4)

if __name__ == "__main__":
    main(EXPERIMENT)
