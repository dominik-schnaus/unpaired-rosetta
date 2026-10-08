"""Shared geometry predicts alignment across modalities (Sec. 4.3): Fig. 4b, Table 5 and Table 12.

Aligns eleven modality pairs from the natural sciences and plots our FOSCTTM against the CKA of the paired
embeddings, together with the benchmarks of Sec. 4.2. Each item is on one training side only, and evaluation uses all
paired items, including the training items.

Writes ``fig4b_foscttm_vs_cka.png`` with its fit statistics, ``tab5_setup.{png,md}`` and
``tab12_other_domains.{png,csv,md}`` to ``$UNPAIRED_ROSETTA_ROOT/figures/other_domains/``.

    python -m experiments.other_domains run [--workers 8 --threads 4]
    python -m experiments.other_domains plot
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from experiments import plotting, shared_geometry
from experiments.alignment import AlignmentRun, run_alignment
from experiments.runner import Experiment, figures_dir, main, results_dir
from experiments.shared_geometry import GeometryRun
from modalities import (
    galaxy_spectra,
    histology_expression,
    interatomic_potentials,
    meg_text,
    msms_molecules,
    perturbation_profiles,
    single_cell,
    tissue_ct,
    tissue_mri,
)
from modalities.common import EmbeddingFile
from unpaired_rosetta.randomness import SEEDS
from unpaired_rosetta.results import load_results

if TYPE_CHECKING:
    from matplotlib.axes import Axes

DOMAIN_SPECIFIC_SEED = 42  # geometry scores of NQ, PBMC and NSD do not depend on a run
SCORES = {"cka": "CKA", "mutual_knn": "mutual k-NN", "tsi": "TSI"}


@dataclass(frozen=True)
class Pair:
    name: str  # as in Tables 5 and 12
    label: str  # point label in Fig. 4b
    domain: str
    x: EmbeddingFile
    y: EmbeddingFile
    # The sides hold different items. For human and mouse, row i of both sides is a cell of the same type.
    separate_corpora: bool = False

    def run(
        self,
        seed: int,
    ) -> AlignmentRun:
        return AlignmentRun(
            self.x.dataset,
            self.x.model,
            self.y.model,
            seed,
            "ours",
            modality_x=self.x.modality,
            modality_y=self.y.modality,
            # Separate corpora: draw both training sets independently.
            dataset_y=self.y.dataset if self.separate_corpora else None,
            validation=self.x.dataset,
        )


PAIRS = [  # in the order of Table 5
    Pair("MLIP ↔ MLIP (MP-20)", "MLIP ↔ MLIP", "materials science", *interatomic_potentials.PAIR),
    Pair(
        "human ↔ mouse scRNA",
        "human ↔ mouse scRNA",
        "biology",
        *single_cell.embedding_files("CrossSpecies-SC"),
        separate_corpora=True,
    ),
    Pair(
        "scRNA ↔ scATAC (SNARE-seq)", "scRNA ↔ scATAC (SNARE-seq)", "biology", *single_cell.embedding_files("SNARE-seq")
    ),
    Pair("histology ↔ expression (HEST CCRCC)", "histology ↔ expression", "medicine", *histology_expression.PAIR),
    Pair("scRNA ↔ ADT (CITE-seq)", "scRNA ↔ ADT", "biology", *single_cell.embedding_files("CITE-seq")),
    Pair("tissue ↔ CT (CPTAC)", "tissue ↔ CT", "medicine", *tissue_ct.PAIR),
    Pair("MEG ↔ text (SpanishBCBL)", "MEG ↔ text", "neuroscience", *meg_text.PAIR),
    Pair("tissue ↔ MRI (TCGA glioma)", "tissue ↔ MRI", "medicine", *tissue_mri.PAIR),
    Pair("Cell Painting ↔ L1000 (Rosetta)", "Cell Painting ↔ L1000", "biology", *perturbation_profiles.PAIR),
    Pair("MS/MS ↔ molecules (MassSpecGym)", "MS/MS ↔ molecules", "chemistry", *msms_molecules.PAIR),
    Pair("galaxy ↔ spectrum (DESI)", "galaxy ↔ spectrum", "astronomy", *galaxy_spectra.PAIR),
]
# Benchmarks of Sec. 4.2 by training corpus: (name, label, domain).
DOMAIN_SPECIFIC = {
    "nq": ("NQ text ↔ text", "NQ text ↔ text", "language"),
    "PBMC": ("scRNA ↔ scATAC (PBMC)", "scRNA ↔ scATAC (PBMC)", "biology"),
    "NSD-private": ("fMRI ↔ fMRI (NSD subjects)", "fMRI ↔ fMRI", "neuroscience"),
}
DOMAIN_ORDER = ["language", "biology", "medicine", "neuroscience", "chemistry", "materials science", "astronomy"]
# Identifies a set of paired embeddings.
GEOMETRY_KEY = ["scored_corpus", "model_x", "model_y", "modality_x", "modality_y"]


def domain_specific_alignment_runs() -> list[AlignmentRun]:
    """Our runs of ``experiments/domain_specific.py`` on NQ, PBMC and NSD."""
    from experiments import domain_specific

    return [
        run
        for run in domain_specific.EXPERIMENT.configurations()
        if run.method == "ours" and run.dataset in DOMAIN_SPECIFIC
    ]


def domain_specific_geometry_runs() -> list[GeometryRun]:
    """One geometry run per set of paired validation embeddings of the Sec. 4.2 benchmarks."""
    runs = {
        GeometryRun(
            run.validation,
            run.model_x,
            run.model_y,
            run.modality_x,
            run.modality_y,
            run.holdout,
            seed=DOMAIN_SPECIFIC_SEED,
        )
        for run in domain_specific_alignment_runs()
    }
    return sorted(runs, key=str)


def configurations() -> list:
    return [pair.run(seed) for pair in PAIRS for seed in SEEDS] + domain_specific_geometry_runs()


def run(
    config: AlignmentRun | GeometryRun,
    num_workers: int,
    num_threads: int,
) -> dict:
    if isinstance(config, GeometryRun):
        return shared_geometry.run(config)
    _, metrics = run_alignment(config, num_workers, num_threads)
    scored = GeometryRun(
        config.validation, config.model_x, config.model_y, config.modality_x, config.modality_y, seed=config.seed
    )
    return metrics | shared_geometry.run(scored)


def pair_rows() -> pd.DataFrame:
    """One row per run of the eleven pairs, with its FOSCTTM and geometry scores."""
    frame = load_results(results_dir("other_domains"))
    frame = frame[frame["method"] == "ours"]
    names = {(pair.x.dataset, pair.x.model, pair.y.model, pair.x.modality, pair.y.modality): pair for pair in PAIRS}
    keys = zip(frame["dataset"], frame["model_x"], frame["model_y"], frame["modality_x"], frame["modality_y"])
    pairs = [names.get(key) for key in keys]
    return frame.assign(
        pair=[pair.name for pair in pairs], label=[pair.label for pair in pairs], domain=[pair.domain for pair in pairs]
    )


def domain_specific_rows() -> pd.DataFrame:
    """One row per run of our aligner on NQ, PBMC and NSD, with the geometry scores of its validation set."""
    alignment = load_results(results_dir("domain_specific"))
    if alignment.empty:  # domain_specific has not run yet, so show the eleven pairs alone
        return alignment
    alignment = alignment[(alignment["method"] == "ours") & alignment["dataset"].isin(DOMAIN_SPECIFIC)]
    alignment = alignment.drop(columns=[column for column in SCORES if column in alignment]).assign(
        scored_corpus=alignment["validation"]
    )
    geometry = load_results(results_dir("other_domains"))
    geometry = geometry[geometry["method"].isna()].rename(columns={"dataset": "scored_corpus"})
    joined = alignment.merge(geometry[GEOMETRY_KEY + list(SCORES)], on=GEOMETRY_KEY, how="inner")
    names = joined["dataset"].map(DOMAIN_SPECIFIC)
    return joined.assign(pair=names.str[0], label=names.str[1], domain=names.str[2])


def all_rows() -> pd.DataFrame:
    frame = pd.concat([pair_rows(), domain_specific_rows()], ignore_index=True)
    frame["domain"] = pd.Categorical(frame["domain"], DOMAIN_ORDER)
    return frame.sort_values(["domain", "pair"])


def markdown(
    table: pd.DataFrame,
) -> str:
    lines = ["| " + " | ".join(map(str, table.columns)) + " |", "|" + " --- |" * table.shape[1]]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def table12(
    frame: pd.DataFrame,
) -> None:
    """Mean +- std over the runs of each pair, across seeds and encoder or subject pairs."""
    columns = {"foscttm": "FOSCTTM ↓", **{score: f"{label} ↑" for score, label in SCORES.items()}}
    frame = frame.assign(domain=frame["domain"].astype(str))  # sort domains alphabetically
    grouped = frame.groupby(["domain", "pair"], sort=True)[list(columns)]
    means, stds = grouped.mean(), grouped.std()
    table = pd.DataFrame(index=means.index)
    for metric, header in columns.items():
        table[header] = [
            plotting.format_mean_std(mean, std, digits=2) for mean, std in zip(means[metric], stds[metric])
        ]
    table = table.reset_index().rename(columns={"domain": "Domain", "pair": "Pair"})
    output = figures_dir("other_domains") / "tab12_other_domains"
    numbers = means.join(stds, lsuffix="_mean", rsuffix="_std").join(grouped.size().rename("runs")).reset_index()
    plotting.save(plotting.table_figure(table, "Table 12: all fourteen modality pairs"), output, numbers)
    output.with_suffix(".md").write_text(markdown(table))
    print(table.to_string(index=False))


def label_points(
    axis: Axes,
    points: pd.DataFrame,
    score: str,
) -> None:
    """Name each point. A label that would overlap the previous one moves down a line."""
    previous, offset = None, 2
    for _, point in points.sort_values(score).iterrows():
        position = (point[score], point["foscttm"])
        close = previous is not None and abs(position[0] - previous[0]) < 0.08 and abs(position[1] - previous[1]) < 0.03
        offset = offset - 7 if close else 2
        axis.annotate(point["label"], position, fontsize=5, xytext=(3, offset), textcoords="offset points")
        previous = position


def figure4b(
    frame: pd.DataFrame,
) -> None:
    """FOSCTTM against CKA with one point per pair (mean over its runs) and a least-squares fit."""
    points = frame.groupby(["domain", "pair", "label"], observed=True)[["foscttm", *SCORES]].mean().reset_index()
    statistics = []
    for score, label in SCORES.items():
        fit = plotting.linear_fit(points[score], points["foscttm"], points["pair"])
        statistics.append(
            {
                "score": score,
                "r2": fit["r2"],
                "spearman": fit["spearman"],
                "pearson": fit["pearson"],
                "slope": fit["slope"],
                "points": fit["num_points"],
            }
        )
        if score != "cka":
            continue
        figure, axis = plt.subplots(figsize=(6.2, 3.6))
        plotting.fit_scatter(axis, points, score, "foscttm", "domain", fit)
        label_points(axis, points, score)
        axis.set_xlim(0, 1)
        axis.set_xlabel(label)
        axis.set_ylabel("FOSCTTM (ours)")
        axis.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        plotting.save(figure, figures_dir("other_domains") / "fig4b_foscttm_vs_cka", points)
    statistics = pd.DataFrame(statistics)
    statistics.to_csv(figures_dir("other_domains") / "fig4b_fit_statistics.csv", index=False)
    print(statistics.to_string(index=False))


def table5() -> None:
    """Setup of the eleven pairs: the encoder of each side and the number of paired validation samples."""
    table = pd.DataFrame(
        {
            "Pair": [pair.name for pair in PAIRS],
            "Encoder 1": [pair.x.model for pair in PAIRS],
            "Encoder 2": [pair.y.model for pair in PAIRS],
            "Val.": [f"{pair.x.shape[0]:,}" for pair in PAIRS],
        }
    )
    output = figures_dir("other_domains") / "tab5_setup"
    plotting.save(plotting.table_figure(table, "Table 5: the modality pairs of Sec. 4.3"), output)
    output.with_suffix(".md").write_text(markdown(table))


def plot() -> None:
    table5()
    frame = all_rows()
    table12(frame)
    figure4b(frame)


EXPERIMENT = Experiment("other_domains", configurations, run, plot, memory_gb=64, hours=12)

if __name__ == "__main__":
    main(EXPERIMENT)
