"""Qualitative figures: Fig. 19, what a good, a mediocre and a failed alignment look like (App. C.4).

Refits the median-seed run of three pairs from ``experiments/other_domains.py``. The fit is deterministic, so it
matches the stored row. Each panel shows a joint PCA of both sides and how often nearest neighbours share a label the
aligner never saw.

Writes ``fig19_domains.png`` and ``fig19_domains.csv`` to ``$UNPAIRED_ROSETTA_ROOT/figures/qualitative/``.

    python -m experiments.qualitative run    # needs the five seeds of the three pairs in results/other_domains
    python -m experiments.qualitative plot
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch import Tensor
from torch.nn.functional import normalize

from experiments import plotting
from experiments.alignment import run_alignment
from experiments.other_domains import PAIRS
from experiments.runner import Experiment, figures_dir, main, results_dir
from modalities import galaxy_spectra
from unpaired_rosetta.embeddings import embedding_path, load_rows, storage_root
from unpaired_rosetta.linalg import center_and_normalize
from unpaired_rosetta.randomness import SEEDS, Randomness
from unpaired_rosetta.results import load_results

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from experiments.other_domains import Pair
    from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

# (pair, title, subtitle), from good to failed
PANELS = [
    ("scRNA ↔ scATAC (SNARE-seq)", "scRNA ↔ scATAC", "SNARE-seq"),
    ("human ↔ mouse scRNA", "human ↔ mouse", "scRNA, CELLxGENE"),
    ("galaxy ↔ spectrum (DESI)", "galaxy ↔ spectrum", "DESI"),
]
POINTS = 2000  # items per side in a cloud
SNARE_SEQ_CELL_LINES = ["H1", "GM", "BJ", "K562"]  # label i is LabelID i + 1 in SCOT's SNAREseq_metadata.txt
LINEAGES = Path(__file__).with_name("qualitative_labels") / "cross_species_lineages.json"
SCORES = {"foscttm": "FOSCTTM", "cka": "CKA", "mutual_knn": "mutual k-NN", "tsi": "TSI"}


@dataclass(frozen=True)
class DomainPanel:
    pair: str  # a pair of Table 12
    seed: int  # median seed of its five runs


def pair_of(
    name: str,
) -> Pair:
    return next(pair for pair in PAIRS if pair.name == name)


def median_seed(
    frame: pd.DataFrame,
    name: str,
) -> int | None:
    """The seed with the median FOSCTTM of a pair's five runs, or None until all five exist."""
    pair = pair_of(name)
    runs = frame[
        (frame["dataset"] == pair.x.dataset)
        & (frame["modality_x"] == pair.x.modality)
        & (frame["modality_y"] == pair.y.modality)
        & (frame["method"] == "ours")
        & frame["seed"].isin(SEEDS)
    ]
    if len(runs) < len(SEEDS):
        return None
    return int(runs.sort_values("foscttm").iloc[len(runs) // 2]["seed"])


def configurations() -> list[DomainPanel]:
    frame = load_results(results_dir("other_domains"))
    seeds = {name: median_seed(frame, name) if len(frame) else None for name, _, _ in PANELS}
    missing = [name for name, seed in seeds.items() if seed is None]
    if missing:
        print(f"Fig. 19 needs all five runs of {missing} in results/other_domains first.")
    return [DomainPanel(name, seed) for name, seed in seeds.items() if seed is not None]


def labels_of(
    name: str,
) -> tuple[Tensor, list[str], bool]:
    """One label per item (the same on both sides), the label names, and whether the labels are ordered."""
    if "SNARE-seq" in name:
        return torch.load(storage_root() / "embeddings" / "SNARE-seq" / "labels.pt"), SNARE_SEQ_CELL_LINES, False
    if name.startswith("human"):
        lineages = json.loads(LINEAGES.read_text())["lineages"]
        lineage_of_type = torch.empty(sum(len(types) for types in lineages.values()), dtype=torch.long)
        for index, types in enumerate(lineages.values()):
            lineage_of_type[torch.tensor(types)] = index
        cell_types = torch.load(storage_root() / "embeddings" / "CrossSpecies-SC" / "labels_human.pt")
        return lineage_of_type[cell_types], list(lineages), False
    deciles, names = galaxy_spectra.redshift_deciles()
    return deciles, names, True


def joint_pca(
    samples_x: Tensor,
    samples_y: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Both sides on the unit sphere, projected onto the top two principal axes of their union.

    Also returns the fraction of variance each axis explains.
    """
    samples_x, samples_y = normalize(samples_x, dim=-1), normalize(samples_y, dim=-1)
    joint = torch.cat([samples_x, samples_y]).double()
    mean = joint.mean(dim=0, keepdim=True)
    values, vectors = torch.linalg.eigh((joint - mean).T @ (joint - mean))
    basis = vectors[:, -2:].flip(-1)
    explained = values.flip(0)[:2] / values.clamp(min=0).sum()
    return (
        ((samples_x.double() - mean) @ basis).float(),
        ((samples_y.double() - mean) @ basis).float(),
        explained.float(),
    )


def nearest_labels(
    queries: Tensor,
    keys: Tensor,
    labels: Tensor,
    block_size: int = 2048,
) -> Tensor:
    """Label of the most similar key for each query. Both must be in one space."""
    return torch.cat(
        [
            labels[(queries[start : start + block_size] @ keys.T).argmax(dim=1)]
            for start in range(0, len(queries), block_size)
        ]
    )


def within_group_foscttm(
    mapped_x: Tensor,
    keys_y: Tensor,
    groups: Tensor,
    block_size: int = 2048,
) -> float:
    """FOSCTTM when only keys from the query's own group are candidates."""
    fractions = []
    for start in range(0, len(mapped_x), block_size):
        stop = min(start + block_size, len(mapped_x))
        scores = mapped_x[start:stop] @ keys_y.T
        same = groups[start:stop, None] == groups[None, :]
        truth = scores[torch.arange(stop - start), torch.arange(start, stop)]
        closer = ((scores > truth[:, None]) & same).sum(dim=1).float()
        fractions.append(closer / (same.sum(dim=1) - 1).clamp(min=1).float())
    return torch.cat(fractions).mean().item()


def run(
    config: DomainPanel,
    num_workers: int,
    num_threads: int,
) -> dict:
    aligner, _ = run_alignment(pair_of(config.pair).run(config.seed), num_workers, num_threads)
    return panel_data(config, aligner)


@torch.inference_mode()
def panel_data(
    config: DomainPanel,
    aligner: WassersteinProcrustes,
) -> dict:
    """The data of one panel: the projected clouds, the agreement matrix and its summary numbers."""
    pair = pair_of(config.pair)
    side_x = load_rows(embedding_path(pair.x.dataset, pair.x.modality, pair.x.model))
    side_y = load_rows(embedding_path(pair.y.dataset, pair.y.modality, pair.y.model))
    mapped_x = aligner.transform(side_x)  # X in the centered, normalized coordinates of Y
    keys_y = center_and_normalize(side_y, aligner.mean_y)
    labels, names, ordered = labels_of(config.pair)
    shown = Randomness(config.seed).subset(len(side_x), POINTS)
    points_x, points_y, explained = joint_pca(mapped_x[shown], keys_y[shown])
    forward = nearest_labels(mapped_x, keys_y, labels)
    backward = nearest_labels(keys_y @ aligner.weight.T, center_and_normalize(side_x, aligner.mean_x), labels)
    matrix = torch.zeros(len(names), len(names))
    matrix.index_put_((labels, forward), torch.ones(len(labels)), accumulate=True)
    return {
        "names": names,
        "ordered": ordered,
        "points_x": points_x.tolist(),
        "points_y": points_y.tolist(),
        "labels": labels[shown].tolist(),
        "explained": explained.tolist(),
        "matrix": (matrix / matrix.sum(dim=1, keepdim=True).clamp(min=1)).tolist(),
        "agreement_forward": (forward == labels).float().mean().item(),
        "agreement_backward": (backward == labels).float().mean().item(),
        "within_decile_foscttm": within_group_foscttm(mapped_x, keys_y, labels) if ordered else None,
    }


def stored_scores(
    frame: pd.DataFrame,
    panel: DomainPanel,
) -> dict:
    pair = pair_of(panel.pair)
    row = frame[
        (frame["dataset"] == pair.x.dataset)
        & (frame["modality_x"] == pair.x.modality)
        & (frame["method"] == "ours")
        & (frame["seed"] == panel.seed)
    ].iloc[0]
    return {score: float(row[score]) for score in SCORES}


def palette(
    names: list[str],
    ordered: bool,
) -> list:
    if ordered:
        ramp = LinearSegmentedColormap.from_list("deciles", ["#98C6EA", "#003359"])
        return [ramp(index / max(len(names) - 1, 1)) for index in range(len(names))]
    return [plotting.PALETTE[index % len(plotting.PALETTE)] for index in range(len(names))]


def cloud(
    axis: Axes,
    panel: dict,
    pair: Pair,
    title: str,
    subtitle: str,
    scores: dict,
) -> None:
    labels, colors = np.array(panel["labels"]), palette(panel["names"], panel["ordered"])
    points_x, points_y = np.array(panel["points_x"]), np.array(panel["points_y"])
    for index in range(len(panel["names"])):
        shown = labels == index
        axis.scatter(*points_x[shown].T, s=5, marker="o", color=colors[index], alpha=0.4, linewidths=0)
        axis.scatter(*points_y[shown].T, s=7, marker="^", color=colors[index], alpha=0.4, linewidths=0)
    handles = [
        plt.Line2D([], [], marker=marker, linestyle="none", color="#666666", label=modality)
        for marker, modality in (("o", pair.x.modality), ("^", pair.y.modality))
    ]
    axis.legend(handles=handles, loc="upper left", fontsize=6.5)
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(f"{title}\n{subtitle}", fontsize=8.5)
    text = "   ".join(f"{label} {scores[score]:.2f}" for score, label in SCORES.items())
    axis.text(
        0.98,
        0.02,
        text,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
    )


def agreement_matrix(
    axis: Axes,
    panel: dict,
) -> None:
    ramp = LinearSegmentedColormap.from_list("agreement", ["#FFFFFF", plotting.OURS_COLOR])
    axis.imshow(np.array(panel["matrix"]), cmap=ramp, vmin=0, vmax=1, interpolation="nearest")
    axis.grid(False)
    names = panel["names"]
    axis.set_xticks(range(len(names)), names, fontsize=5.5, rotation=90)
    axis.set_yticks(range(len(names)), names, fontsize=5.5)
    axis.set_xlabel("label of the nearest neighbour", fontsize=7)
    axis.set_ylabel("label of the query", fontsize=7)
    title = f"1-NN agreement {100 * panel['agreement_forward']:.0f} % / {100 * panel['agreement_backward']:.0f} %"
    if panel["within_decile_foscttm"] is not None:
        title += f"\nFOSCTTM {panel['within_decile_foscttm']:.2f} within a decile"
    axis.set_title(title, fontsize=7.5)


def figure19() -> None:
    frame = load_results(results_dir("other_domains"))
    panels = {}
    for path in sorted(results_dir("qualitative").glob("*.json")):
        record = json.loads(path.read_text())
        panels[record["config"]["pair"]] = (DomainPanel(**record["config"]), record["metrics"])
    figure, axes = plt.subplots(2, len(PANELS), figsize=(3.3 * len(PANELS), 6.2), squeeze=False)
    numbers = []
    for column, (name, title, subtitle) in enumerate(PANELS):
        if name not in panels:
            continue
        config, panel = panels[name]
        scores = stored_scores(frame, config)
        cloud(axes[0, column], panel, pair_of(name), title, subtitle, scores)
        agreement_matrix(axes[1, column], panel)
        numbers.append(
            {
                "pair": name,
                "seed": config.seed,
                **scores,
                "agreement_forward": panel["agreement_forward"],
                "agreement_backward": panel["agreement_backward"],
                "within_decile_foscttm": panel["within_decile_foscttm"],
                "pc1_explained": panel["explained"][0],
                "pc2_explained": panel["explained"][1],
            }
        )
    figure.tight_layout()
    plotting.save(figure, figures_dir("qualitative") / "fig19_domains", pd.DataFrame(numbers))


def plot() -> None:
    figure19()


EXPERIMENT = Experiment("qualitative", configurations, run, plot, memory_gb=64, hours=12)

if __name__ == "__main__":
    main(EXPERIMENT)
