"""Shared parts of the alignment ablations in Appendix A.2: the three settings, the run configuration, and one run.

The settings are MS COCO (DINOv2 and Qwen3-8B), NQ (GTR and GTE) and SNARE-seq (RNA and ATAC).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pandas as pd

from experiments.ablation_figures import DATASET_LABELS, mean_std
from experiments.alignment import AlignmentRun, evaluate, load_training_data, load_validation_data
from unpaired_rosetta.ablation.aligner import AblatedWassersteinProcrustes
from unpaired_rosetta.randomness import SEEDS, Randomness

if TYPE_CHECKING:
    from torch import Tensor

QWEN3_GEN = "qwen3-8b-gen_69445574b3021b8a2733e87d8603d769"
SETTINGS = {
    "nq": AlignmentRun(
        dataset="nq",
        model_x="gtr",
        model_y="gte",
        seed=0,
        modality_x="language",
        modality_y="language",
        validation="nq",
        holdout=8192,
        train_size=250_000,
    ),
    "SNARE-seq": AlignmentRun(
        dataset="SNARE-seq",
        model_x="pca10",
        model_y="lsi19",
        seed=0,
        modality_x="rna",
        modality_y="atac",
        validation="SNARE-seq",
    ),
    "coco_train2014": AlignmentRun(
        dataset="coco_train2014",
        model_x="dinov2_vit-b14@224_mean",
        model_y=QWEN3_GEN,
        seed=0,
        validation="coco_val2014",
    ),
}


@dataclass(frozen=True)
class AblationRun:
    """One ablated aligner on one setting and seed. The hyperparameters default to the paper's."""

    setting: str
    seed: int
    initialization: str = "ours"
    readout: str = "ours"
    refinement: str = "ours"
    num_clusters: int = 30  # C
    num_restarts: int = 30  # S
    batch_size: int = 10_000  # b
    num_iterations: int = 100  # R


def alignment_run(
    config: AblationRun,
) -> AlignmentRun:
    return replace(SETTINGS[config.setting], seed=config.seed)


def build_aligner(
    config: AblationRun,
    num_workers: int = 1,
    num_threads: int = 1,
) -> AblatedWassersteinProcrustes:
    return AblatedWassersteinProcrustes(
        config.initialization,
        config.readout,
        config.refinement,
        num_clusters=config.num_clusters,
        num_restarts=config.num_restarts,
        batch_size=config.batch_size,
        num_iterations=config.num_iterations,
        num_workers=num_workers,
        num_threads=num_threads,
    )


def run_ablation(
    config: AblationRun,
    num_workers: int = 1,
    num_threads: int = 1,
) -> dict:
    """Split the data, fit the ablated aligner and evaluate it, as ``experiments.alignment.run_alignment`` does."""
    run = alignment_run(config)
    randomness = Randomness(run.seed)
    samples_x, samples_y, _, _ = load_training_data(run, randomness)
    aligner = build_aligner(config, num_workers, num_threads)
    aligner.fit(samples_x, samples_y, None, None, randomness)
    return evaluate(aligner, run)


def load_prepared_data(
    config: AblationRun,
    num_workers: int = 1,
    num_threads: int = 1,
) -> tuple[AblatedWassersteinProcrustes, Tensor, Tensor, Randomness, AlignmentRun]:
    """Shared start for runs that compare stages on one initialization.

    Returns the aligner with its preprocessing fitted, the preprocessed training sets, the randomness right after the
    data split, and the run.
    """
    run = alignment_run(config)
    randomness = Randomness(run.seed)
    samples_x, samples_y, _, _ = load_training_data(run, randomness)
    aligner = build_aligner(config, num_workers, num_threads)
    samples_x, samples_y = aligner.preprocess(samples_x, samples_y)
    return aligner, samples_x, samples_y, randomness, run


def evaluate_variants(
    aligner: AblatedWassersteinProcrustes,
    run: AlignmentRun,
    weights: dict,
) -> dict:
    """Metrics named ``<variant>/<metric>`` for the aligner with each given map."""
    metrics = {}
    for variant, weight in weights.items():
        aligner.weight = weight
        metrics.update({f"{variant}/{key}": value for key, value in evaluate(aligner, run).items()})
    return metrics


def validation_after_split(
    run: AlignmentRun,
    randomness: Randomness,
) -> tuple[Tensor, Tensor]:
    """The validation set in the order drawn right after the data split. ``randomness`` is not advanced."""
    val_x, val_y, _ = load_validation_data(run, copy.deepcopy(randomness))
    return val_x, val_y


def configurations(
    **variants: list[object],
) -> list[AblationRun]:
    """One run per setting, seed and combination of the given ``field=[values]``."""
    grid = [{}]
    for field, values in variants.items():
        grid = [{**point, field: value} for point in grid for value in values]
    return [AblationRun(setting, seed, **point) for setting in SETTINGS for seed in SEEDS for point in grid]


def stage_table(
    frame: pd.DataFrame,
    rows: dict,
    column: str,
) -> pd.DataFrame:
    """FOSCTTM as ``mean +- std`` over the seeds, with one row per variant and one column per setting.

    ``rows`` maps each value of ``column`` to its row label.
    """
    table = {}
    for setting, label in DATASET_LABELS.items():
        cells = frame[frame["setting"] == setting]
        table[label] = [
            mean_std(cells[cells[column] == variant]["foscttm"]) if (cells[column] == variant).any() else "--"
            for variant in rows
        ]
    return pd.DataFrame(table, index=pd.Index(list(rows.values()), name=column.capitalize()))


def results_frame(
    results: pd.DataFrame,
    prefix_column: str,
    variants: list[str],
) -> pd.DataFrame:
    """Turn the ``<variant>/foscttm`` columns into one row per variant."""
    parts = []
    for variant in variants:
        column = f"{variant}/foscttm"
        if column in results:
            parts.append(results[["setting", "seed"]].assign(**{prefix_column: variant, "foscttm": results[column]}))
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["setting", "seed", prefix_column, "foscttm"])
    )
