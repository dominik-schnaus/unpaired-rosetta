"""Unpaired alignment across domains (Sec. 4.2): Fig. 4a and Tables 8, 9 and 10.

Compares our aligner with a domain-specific baseline on NQ text encoders (vec2vec, mini-vec2vec), PBMC single-cell
data (SCOT+) and NSD fMRI subjects (the platonic-brain aligner).

The platonic-brain baseline of Marcos-Manchon et al. (2026) aligns two subjects with a mini-vec2vec variant.
``modalities/brain_scans/pipeline`` runs their code ten times for each of the 56 ordered subject pairs, giving one
orthogonal map ``W`` per run and pair, and this script scores each map with our metrics on the 907 shared validation
images as ``normalize((x - mean_x) W)`` against ``normalize(y - mean_y)``. The means come from the validation rows,
because their maps were fitted on standardized embeddings without a mean of their own.

    python -m experiments.domain_specific run --slurm --jobs 30 --workers 4 --threads 4 --only "method != 'vec2vec'"
    python -m experiments.domain_specific run --slurm --jobs 10 --only "method == 'vec2vec'" --gpus 1
    python -m experiments.domain_specific plot
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.nn.functional import normalize

from experiments import plotting
from experiments.alignment import AlignmentRun, run_alignment
from experiments.other_domains import markdown
from experiments.runner import Experiment, figures_dir, main, results_dir
from modalities import brain_scans, natural_questions, single_cell
from unpaired_rosetta.embeddings import embedding_path, load_rows
from unpaired_rosetta.evaluation import retrieval_metrics
from unpaired_rosetta.randomness import SEEDS, Randomness, derive_seeds
from unpaired_rosetta.results import load_results

NSD_SEEDS = derive_seeds(42, 10)
BENCHMARKS = {"nq": "NQ", "PBMC": "PBMC", "NSD-private": "NSD fMRI"}
METHODS = {"NQ": ["vec2vec", "mini-vec2vec", "ours"], "PBMC": ["SCOT+", "ours"], "NSD fMRI": ["platonic brain", "ours"]}
HEADERS = {"foscttm": "FOSCTTM ↓", "mean_rank": "mean rank ↓"}


def nq_runs() -> list[AlignmentRun]:
    return [
        AlignmentRun(
            natural_questions.DATASET,
            x,
            y,
            seed,
            method,
            modality_x="language",
            modality_y="language",
            validation=natural_questions.DATASET,
            holdout=natural_questions.HOLDOUT,
            train_size=natural_questions.TRAIN_SIZE,
        )
        for method in METHODS["NQ"]
        for x, y in natural_questions.PAIRS
        for seed in SEEDS
    ]


def pbmc_runs() -> list[AlignmentRun]:
    (modality_x, model_x), (modality_y, model_y) = single_cell.PAIRS["PBMC"]
    return [
        AlignmentRun(
            "PBMC",
            model_x,
            model_y,
            seed,
            method,
            modality_x=modality_x,
            modality_y=modality_y,
            validation="PBMC",
            labels="labels.pt",
        )
        for method in METHODS["PBMC"]
        for seed in SEEDS
    ]


def nsd_runs() -> list[AlignmentRun]:
    """Our aligner on every ordered subject pair.

    The training sets are the private images of two subjects, so no image is on both sides. ``dataset_y`` makes them
    independent draws instead of halves of one corpus.
    """
    return [
        AlignmentRun(
            brain_scans.TRAINING_DATASET,
            brain_scans.ENCODER,
            brain_scans.ENCODER,
            seed,
            modality_x=x,
            modality_y=y,
            dataset_y=brain_scans.TRAINING_DATASET,
            validation=brain_scans.VALIDATION_DATASET,
        )
        for x, y in brain_scans.SUBJECT_PAIRS
        for seed in NSD_SEEDS
    ]


RUNS = range(1, 11)  # the authors' code uses no seed, so the index only names the run


@dataclass(frozen=True)
class PlatonicBrainRun:
    subject_x: str
    subject_y: str
    run: int


class FixedMap:
    """A fixed map, applied to samples centered with given means. It is never fitted."""

    def __init__(
        self,
        weight: Tensor,
        mean_x: Tensor,
        mean_y: Tensor,
    ) -> None:
        self.weight, self.mean_x, self.mean_y = weight, mean_x, mean_y

    def similarity(
        self,
        queries_x: Tensor,
        keys_y: Tensor,
    ) -> Tensor:
        return normalize((queries_x - self.mean_x) @ self.weight, dim=-1) @ normalize(keys_y - self.mean_y, dim=-1).T


def alignment_file(
    config: PlatonicBrainRun,
) -> Path:
    """The authors' output file for one run and pair. Its name ends with their validation score."""
    folder = brain_scans.pipeline_root() / "runs" / "1mm" / "alignments" / f"final_alignment_{config.run}"
    [path] = folder.glob(f"mini-vec2vec-alignment-sub-{config.subject_x[3:]}-sub-{config.subject_y[3:]}_*.pt")
    return path


def platonic_runs() -> list[PlatonicBrainRun]:
    return [PlatonicBrainRun(x, y, run) for run in RUNS for x, y in brain_scans.SUBJECT_PAIRS]


def score_platonic_map(
    config: PlatonicBrainRun,
) -> dict:
    weight = torch.load(alignment_file(config), map_location="cpu")["W"].float()
    val_x = load_rows(embedding_path(brain_scans.VALIDATION_DATASET, config.subject_x, brain_scans.ENCODER))
    val_y = load_rows(embedding_path(brain_scans.VALIDATION_DATASET, config.subject_y, brain_scans.ENCODER))
    aligner = FixedMap(weight, val_x.mean(dim=0), val_y.mean(dim=0))
    return retrieval_metrics(aligner, val_x, val_y, Randomness(config.run))


def configurations() -> list[AlignmentRun | PlatonicBrainRun]:
    return nq_runs() + pbmc_runs() + nsd_runs() + platonic_runs()


def run(
    config: AlignmentRun | PlatonicBrainRun,
    num_workers: int,
    num_threads: int,
) -> dict:
    if isinstance(config, PlatonicBrainRun):
        return score_platonic_map(config)
    _, metrics = run_alignment(config, num_workers, num_threads)
    return metrics


def all_rows() -> pd.DataFrame:
    """One row per run with its benchmark, method and pair, including the platonic-brain maps."""
    rows = load_results(results_dir("domain_specific"))
    is_platonic = rows["subject_x"].notna()
    platonic, ours = rows[is_platonic].copy(), rows[~is_platonic].copy()
    ours["benchmark"] = ours["dataset"].map(BENCHMARKS)
    ours["pair"] = ours["modality_x"] + " → " + ours["modality_y"]  # NSD subjects
    nq = ours["benchmark"] == "NQ"
    ours.loc[nq, "pair"] = ours["model_x"] + " → " + ours["model_y"]
    ours.loc[ours["benchmark"] == "PBMC", "pair"] = "RNA → ATAC"
    pairs = platonic["subject_x"] + " → " + platonic["subject_y"]
    platonic = platonic.assign(benchmark="NSD fMRI", method="platonic brain", pair=pairs)
    return pd.concat([ours, platonic], ignore_index=True)


def best_at_precision(
    means: pd.Series,
    digits: int,
    lower_is_better: bool = True,
) -> pd.Series:
    """The cells with the best printed value, so that all ties at the printed precision are marked."""
    printed = means.round(digits)
    return printed == (printed.min() if lower_is_better else printed.max())


def save_table(
    table: pd.DataFrame,
    bold: pd.DataFrame,
    numbers: pd.DataFrame,
    name: str,
    title: str,
) -> None:
    output = figures_dir("domain_specific") / name
    plotting.save(plotting.table_figure(table, title, bold), output, numbers)
    output.with_suffix(".md").write_text(markdown(table))
    print(table.to_string(index=False))


def figure4a(
    frame: pd.DataFrame,
) -> None:
    """FOSCTTM of each method on each benchmark, over all pairs and seeds."""
    rows, bold = [], []
    for benchmark, methods in METHODS.items():
        grouped = frame[frame["benchmark"] == benchmark].groupby("method")["foscttm"]
        means, stds = grouped.mean().reindex(methods), grouped.std().reindex(methods)
        best = best_at_precision(means, 4)
        for method in methods:
            printed = plotting.format_mean_std(means[method], stds[method], 4)
            rows.append(
                {
                    "Benchmark": benchmark,
                    "Method": method,
                    "FOSCTTM ↓": printed,
                    "mean": means[method],
                    "std": stds[method],
                    "runs": grouped.size().get(method, 0),
                }
            )
            bold.append([False, False, bool(best[method])])
    numbers = pd.DataFrame(rows)
    table = numbers[["Benchmark", "Method", "FOSCTTM ↓"]]
    save_table(table, pd.DataFrame(bold), numbers, "fig4a_domain_specific", "Fig. 4a: domain-specific benchmarks")


def pair_table(
    frame: pd.DataFrame,
    benchmark: str,
    digits: dict[str, int],
    name: str,
    title: str,
) -> None:
    """Tables 8 and 10: one row per pair with the mean over seeds of each metric and method. The best is bold."""
    rows = frame[frame["benchmark"] == benchmark]
    grouped = rows.groupby(["pair", "method"])[list(digits)]
    means, stds = grouped.mean(), grouped.std()
    pairs = sorted(rows["pair"].unique())
    table, bold = pd.DataFrame({"Pair": pairs}), pd.DataFrame({"Pair": [False] * len(pairs)})
    for method in METHODS[benchmark]:
        for metric, decimals in digits.items():
            column = f"{method} {HEADERS[metric]}"
            values = [means[metric].get((pair, method), float("nan")) for pair in pairs]
            table[column] = [f"{value:.{decimals}f}" for value in values]
            bold[column] = False
    for metric, decimals in digits.items():
        by_method = means[metric].unstack("method").reindex(index=pairs, columns=METHODS[benchmark])
        best = by_method.round(decimals).eq(by_method.round(decimals).min(axis=1), axis=0)
        for method in METHODS[benchmark]:
            bold[f"{method} {HEADERS[metric]}"] = best[method].to_numpy()
    numbers = means.join(stds, lsuffix="_mean", rsuffix="_std").join(grouped.size().rename("runs")).reset_index()
    save_table(table, bold, numbers, name, title)


def table9(
    frame: pd.DataFrame,
) -> None:
    """PBMC: FOSCTTM and label transfer accuracy of each method, mean +- std over the seeds."""
    rows = frame[frame["benchmark"] == "PBMC"]
    metrics = {"foscttm": ("FOSCTTM ↓", True), "label_transfer_accuracy": ("LTA ↑", False)}
    grouped = rows.groupby("method")[list(metrics)]
    means, stds = grouped.mean().reindex(METHODS["PBMC"]), grouped.std().reindex(METHODS["PBMC"])
    table, bold = pd.DataFrame({"Method": METHODS["PBMC"]}), pd.DataFrame({"Method": [False] * len(METHODS["PBMC"])})
    for metric, (header, lower_is_better) in metrics.items():
        table[header] = [plotting.format_mean_std(mean, std, 3) for mean, std in zip(means[metric], stds[metric])]
        bold[header] = best_at_precision(means[metric], 3, lower_is_better).to_numpy()
    numbers = means.join(stds, lsuffix="_mean", rsuffix="_std").join(grouped.size().rename("runs")).reset_index()
    save_table(table, bold, numbers, "tab9_pbmc", "Table 9: PBMC, RNA → ATAC")


def plot() -> None:
    frame = all_rows()
    figure4a(frame)
    pair_table(frame, "NQ", {"foscttm": 4, "mean_rank": 1}, "tab8_nq", "Table 8: NQ, per encoder pair")
    table9(frame)
    pair_table(frame, "NSD fMRI", {"foscttm": 3, "mean_rank": 1}, "tab10_nsd", "Table 10: NSD fMRI, per subject pair")


EXPERIMENT = Experiment("domain_specific", configurations, run, plot, memory_gb=32, hours=12)

if __name__ == "__main__":
    main(EXPERIMENT)
