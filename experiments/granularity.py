"""Granularity of the alignment (App. E): Fig. 22 (clusters), Fig. 23 (principal subspaces), Fig. 24 (clipping).

Tests at which coarseness two embedding spaces agree, using cluster centers, principal subspaces and clipped
kernels. Each comparison scores the same number of points with the same ``k``, because all scores depend on them.

    python -m experiments.granularity run --slurm --jobs 12
    python -m experiments.granularity plot
"""

import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import ScalarFormatter
from torch import Tensor
from torch.nn.functional import normalize

from experiments import plotting
from experiments.runner import Experiment, figures_dir, main, results_dir
from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.embeddings import embedding_path, load_rows, num_rows
from unpaired_rosetta.geometry import (
    cka,
    cka_of_kernels,
    kernel_neighbors,
    mutual_knn,
    neighbor_overlap,
    quadruplet_similarity_index,
    quadruplet_similarity_index_of_kernels,
    triplet_similarity_index,
    triplet_similarity_index_of_kernels,
)
from unpaired_rosetta.linalg import polar
from unpaired_rosetta.randomness import Randomness, derive_seeds
from unpaired_rosetta.results import load_results

SEEDS = derive_seeds(42, 10)
VISION = "dinov2_vit-b14@224_mean"
LANGUAGE_MODELS = {"mpnet": "MPNet", "qwen3-embedding-8b": "Qwen3-Embedding-8B"}
DATASETS = {"coco_train2014": "MS COCO", "wit_train": "WIT"}
# The Qwen3-Embedding-8B embeddings of WIT are stored under the 0.6B name. They are 4096-dimensional, and re-embedding
# the first captions with the 8B model reproduces them.
STORED_NAMES = {("wit_train", "qwen3-embedding-8b"): "qwen3-embedding-0.6b"}
NUM_THREADS = 4  # as in the paper's runs, since matrix products change in the last bits with the thread count

# Clusters
LIFT_SIZE = {"coco_train2014": None, "wit_train": 500_000}  # None means the whole corpus
FIT_SIZE = 100_000
CLUSTERS = {"coco_train2014": [10, 20, 50, 100, 200], "wit_train": [10, 20, 50, 100, 200, 400]}
GROUPS_SCORED = 32  # fixed, so that the cost does not grow with C
SEEDING_SIZE = 10_000  # rows the k-means seeding draws from
KMEANS_ITERATIONS, SIZE_PENALTY, TOLERANCE = 100, 1e-3, 1e-4
CHUNK_SIZE = 51_200
LIFT_ROUNDS, LIFT_STEP = 500, 0.01

# Principal subspaces and clipping
SUBSAMPLE = 10_000
RANKS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, None]  # None means the full space
THRESHOLDS = [round(0.1 * step, 1) for step in range(-10, 11)]
SMALL = 4096  # up to this many points, TSI and QSI use a distance matrix

MEASURES = {"cka": "CKA", "mutual_knn": "mutual k-NN", "tsi": "TSI", "qsi": "QSI"}


@dataclass(frozen=True)
class GranularityRun:
    analysis: str  # "clusters", "subspaces" or "clipping"
    dataset: str
    language_model: str
    seed: int
    vision_model: str = VISION


def configurations() -> list[GranularityRun]:
    return [
        GranularityRun(analysis, dataset, language, seed)
        for analysis in ["clusters", "subspaces", "clipping"]
        for dataset in DATASETS
        for language in LANGUAGE_MODELS
        for seed in SEEDS
    ]


def neighbors_for(
    num_points: int,
) -> int:
    """``k`` for mutual k-NN at the ratio n / 100 (Koepke et al., 2026), at least one."""
    return max(1, num_points // 100)


def load_pairs(
    config: GranularityRun,
    size: int | None,
    randomness: Randomness,
) -> tuple[Tensor, Tensor]:
    """``size`` random paired items, or all items in random order if None."""
    path_x = embedding_path(config.dataset, "vision", config.vision_model)
    stored_name = STORED_NAMES.get((config.dataset, config.language_model), config.language_model)
    path_y = embedding_path(config.dataset, "language", stored_name)
    indices = randomness.subset(num_rows(path_x), size)
    return load_rows(path_x, indices), load_rows(path_y, indices)


def center_and_normalize(
    samples: Tensor,
) -> Tensor:
    return normalize(samples - samples.mean(dim=0), dim=-1)


def alignment_scores(
    samples_x: Tensor,
    samples_y: Tensor,
    generator: torch.Generator,
) -> dict[str, float]:
    """The four scores with k = max(1, n // 100).

    On few points, TSI and QSI use full distance matrices. The result is the same, only cheaper.
    """
    num_points = samples_x.shape[0]
    scores = {
        "cka": cka(samples_x, samples_y),
        "mutual_knn": mutual_knn(samples_x, samples_y, neighbors_for(num_points)),
    }
    if num_points <= SMALL:
        closeness_x = -torch.cdist(samples_x.double(), samples_x.double())
        closeness_y = -torch.cdist(samples_y.double(), samples_y.double())
        scores["tsi"] = triplet_similarity_index_of_kernels(closeness_x, closeness_y, generator)
        scores["qsi"] = quadruplet_similarity_index_of_kernels(closeness_x, closeness_y, generator)
    else:
        scores["tsi"] = triplet_similarity_index(samples_x, samples_y, generator)
        scores["qsi"] = quadruplet_similarity_index(samples_x, samples_y, generator)
    return scores


def prefixed(
    prefix: str,
    scores: dict[str, float],
) -> dict[str, float]:
    return {f"{prefix}/{name}": value for name, value in scores.items()}


def seed_centroids(
    data: Tensor,
    num_clusters: int,
    generator: torch.Generator,
) -> Tensor:
    """k-means++-like seeding on ``SEEDING_SIZE`` rows drawn with replacement.

    Each next center is drawn with probability proportional to its distance, not the squared distance, to the closest
    center so far.
    """
    data = data[torch.randint(0, data.shape[0], [min(SEEDING_SIZE, data.shape[0])], generator=generator)]
    centroids = torch.zeros((num_clusters, data.shape[1])).to(data)
    centroids[0] = data[torch.randint(data.shape[0], [1], generator=generator)]
    for index in range(1, num_clusters):
        distances = torch.cdist(centroids[:index][None], data[None], p=2)[0].amin(dim=0)
        cumulative = torch.cumsum(distances / distances.sum(), dim=0)
        drawn = torch.searchsorted(cumulative, torch.rand([1], generator=generator))
        centroids[index] = data[drawn.clamp(max=data.shape[0] - 1)]
    return centroids


def nearest_centroids(
    data: Tensor,
    centroids: Tensor,
    offsets: Tensor,
) -> Tensor:
    """``argmin_c ||x - mu_c||^2 + penalty_c`` in chunks of rows. ``offsets`` holds ``||mu_c||^2 + penalty_c``."""
    data_norms = (data**2).sum(dim=1)
    labels = torch.empty(data.shape[0], dtype=torch.long)
    for start in range(0, data.shape[0], CHUNK_SIZE):
        chunk = data[start : start + CHUNK_SIZE]
        distances = data_norms[start : start + CHUNK_SIZE, None] + offsets[None, :]
        distances = distances.addmm_(chunk, centroids.T, alpha=-2.0, beta=1.0)
        labels[start : start + CHUNK_SIZE] = distances.argmin(dim=1)
    return labels


def balanced_kmeans(
    data: Tensor,
    num_clusters: int,
    generator: torch.Generator,
) -> Tensor:
    """Unit-norm centroids of k-means with a size penalty that pushes clusters towards equal sizes.

    The penalty grows with cluster size and with the iteration, as in Malinen & Franti (2014).
    """
    num_samples = data.shape[0]
    centroids = normalize(seed_centroids(data, num_clusters, generator), dim=1)
    relative_sizes = torch.full((num_clusters,), 1.0 / num_clusters, dtype=data.dtype)
    size_penalty = torch.zeros(num_clusters, dtype=data.dtype)
    for iteration in range(KMEANS_ITERATIONS):
        labels = nearest_centroids(data, centroids, (centroids**2).sum(dim=1) + size_penalty)
        counts = torch.bincount(labels, minlength=num_clusters).to(data.dtype)
        sums = torch.zeros_like(centroids).index_add_(0, labels, data)
        # Sort by size, so that each cluster's penalty follows its rank across iterations.
        order = torch.argsort(counts)
        centroids, sums, counts = centroids[order], sums[order], counts[order]
        new_centroids = torch.zeros_like(centroids)
        non_empty = counts > 0
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty, None]
        empty = (~non_empty).nonzero(as_tuple=True)[0]
        if len(empty) > 0:
            new_centroids[empty] = data[torch.randint(0, num_samples, (len(empty),), generator=generator)]
        new_centroids = normalize(new_centroids, dim=1)
        shift = (new_centroids - centroids).norm(dim=1).sum().item()
        centroids = new_centroids
        size_penalty = (
            (iteration + 1) * SIZE_PENALTY * (2 * counts - 2 * num_samples * relative_sizes + 1) / KMEANS_ITERATIONS
        )
        if shift < TOLERANCE:
            break
    return centroids


def lift(
    data: Tensor,
    centroids: Tensor,
    min_size: int,
) -> Tensor:
    """Assign each row to its nearest center, so that every cluster gets at least ``min_size`` members.

    Clusters that are too small get a growing distance discount for at most ``LIFT_ROUNDS`` rounds. Only these clusters
    get a discount, so as few rows as possible leave their nearest center.
    """
    num_clusters = centroids.shape[0]
    norms = (centroids**2).sum(dim=1)
    distances = torch.empty(data.shape[0], num_clusters)
    for start in range(0, data.shape[0], 65_536):
        chunk = data[start : start + 65_536]
        distances[start : start + 65_536] = (chunk**2).sum(dim=1, keepdim=True) + norms - 2 * chunk @ centroids.T
    discount = torch.zeros(num_clusters)
    for _ in range(LIFT_ROUNDS):
        labels = (distances - discount).argmin(dim=1)
        deficit = (min_size - torch.bincount(labels, minlength=num_clusters)).clamp(min=0)
        if int(deficit.sum()) == 0:
            break
        # The constant term keeps the last rows moving. A step that only follows the deficit stalls one row short.
        discount += (deficit > 0).to(discount) * (0.001 + LIFT_STEP * deficit.to(discount) / min_size)
    return labels


def run_clusters(
    config: GranularityRun,
    randomness: Randomness,
) -> dict:
    vision, language = load_pairs(config, LIFT_SIZE[config.dataset], randomness)
    vision, language = center_and_normalize(vision), center_and_normalize(language)
    joint = torch.cat([vision, language], dim=1)  # both halves have unit norm, so neither dominates
    num_points = joint.shape[0]
    fit_rows = randomness.subset(num_points, FIT_SIZE)
    metrics = {}
    for num_clusters in CLUSTERS[config.dataset]:
        centroids = balanced_kmeans(joint[fit_rows], num_clusters, randomness.torch)
        labels = lift(joint, centroids, num_clusters)
        smallest = int(torch.bincount(labels, minlength=num_clusters).min())
        metrics[f"C={num_clusters}/smallest_cluster"] = smallest
        if smallest < num_clusters:
            continue  # a cluster has fewer than C members, so skip this C for this seed

        centers_x = normalize(torch.zeros(num_clusters, vision.shape[1]).index_add_(0, labels, vision), dim=-1)
        centers_y = normalize(torch.zeros(num_clusters, language.shape[1]).index_add_(0, labels, language), dim=-1)
        metrics |= prefixed(f"C={num_clusters}/coarse", alignment_scores(centers_x, centers_y, randomness.torch))

        within = []
        for cluster in randomness.subset(num_clusters, GROUPS_SCORED).tolist():
            members = (labels == cluster).nonzero(as_tuple=True)[0]
            members = members[randomness.subset(members.shape[0], num_clusters)]
            within.append(alignment_scores(vision[members], language[members], randomness.torch))
        metrics |= prefixed(
            f"C={num_clusters}/fine", {name: float(np.mean([s[name] for s in within])) for name in MEASURES}
        )

        rows = randomness.subset(num_points, num_clusters)
        metrics |= prefixed(
            f"C={num_clusters}/random", alignment_scores(vision[rows], language[rows], randomness.torch)
        )
    return metrics


def run_subspaces(
    config: GranularityRun,
    randomness: Randomness,
) -> dict:
    vision, language = load_pairs(config, SUBSAMPLE, randomness)
    vision, language = center_and_normalize(vision), center_and_normalize(language)
    # Center again after normalizing, so that the principal directions are those of the scored points.
    vision, language = vision - vision.mean(dim=0), language - language.mean(dim=0)
    basis_x = torch.linalg.svd(vision.double(), full_matrices=False)[2].T.to(vision)
    basis_y = torch.linalg.svd(language.double(), full_matrices=False)[2].T.to(language)
    metrics = {}
    for rank in RANKS:
        if rank is None:
            metrics |= prefixed("p=full/principal", alignment_scores(vision, language, randomness.torch))
            continue
        principal = alignment_scores(vision @ basis_x[:, :rank], language @ basis_y[:, :rank], randomness.torch)
        metrics |= prefixed(f"p={rank}/principal", principal)
        random_x = polar(torch.randn(vision.shape[1], rank, generator=randomness.torch))
        random_y = polar(torch.randn(language.shape[1], rank, generator=randomness.torch))
        metrics |= prefixed(
            f"p={rank}/random", alignment_scores(vision @ random_x, language @ random_y, randomness.torch)
        )
    return metrics


def run_clipping(
    config: GranularityRun,
    randomness: Randomness,
) -> dict:
    vision, language = load_pairs(config, SUBSAMPLE, randomness)
    # The scores use a second random stream from the seed, as in the paper's runs.
    generator = torch.Generator().manual_seed(config.seed)
    units = {"vision": center_and_normalize(vision), "language": center_and_normalize(language)}
    kernels = {space: (unit @ unit.T).double() for space, unit in units.items()}
    num_neighbors = neighbors_for(SUBSAMPLE)
    neighbors = {space: kernel_neighbors(kernel, num_neighbors, generator) for space, kernel in kernels.items()}
    metrics = {}
    for space, other in [("vision", "language"), ("language", "vision")]:
        for side in ["below", "above"]:
            for threshold in THRESHOLDS:
                if (side == "below" and threshold >= 1.0) or (side == "above" and threshold <= -1.0):
                    continue  # constant kernel, where every score is undefined
                clipped = (
                    kernels[space].clamp(min=threshold) if side == "below" else kernels[space].clamp(max=threshold)
                )
                kernel_x, kernel_y = (clipped, kernels[other]) if space == "vision" else (kernels[other], clipped)
                scores = {
                    "cka": cka_of_kernels(kernel_x, kernel_y),
                    "mutual_knn": neighbor_overlap(
                        kernel_neighbors(clipped, num_neighbors, generator), neighbors[other]
                    ),
                    "tsi": triplet_similarity_index_of_kernels(kernel_x, kernel_y, generator),
                    "qsi": quadruplet_similarity_index_of_kernels(kernel_x, kernel_y, generator),
                }
                metrics |= prefixed(f"{space}/{side}/{threshold:+.1f}", scores)
    return metrics


ANALYSES = {"clusters": run_clusters, "subspaces": run_subspaces, "clipping": run_clipping}


def run(
    config: GranularityRun,
    num_workers: int = 1,
    num_threads: int = 1,
) -> dict:
    with torch_threads(NUM_THREADS):
        return ANALYSES[config.analysis](config, Randomness(config.seed))


PANELS = [(dataset, model) for dataset in DATASETS for model in LANGUAGE_MODELS]
BLUE, ORANGE, GREEN = plotting.PALETTE[0], plotting.PALETTE[1], plotting.PALETTE[2]


def melt(
    frame: pd.DataFrame,
    pattern: str,
    names: list[str],
) -> pd.DataFrame:
    """One row per run and metric column matching ``pattern``, with its groups as columns and the score as ``value``.

    Undefined scores (NaN or +-inf) are dropped.
    """
    regex = re.compile(pattern)
    parts = []
    for column in frame.columns:
        match = regex.fullmatch(column)
        if match is None:
            continue
        part = frame[["dataset", "language_model", "seed", column]].replace([np.inf, -np.inf], np.nan)
        part = part.dropna(subset=[column]).rename(columns={column: "value"})
        parts.append(part.assign(**dict(zip(names, match.groups()))))
    return pd.concat(parts, ignore_index=True)


def grid(
    long: pd.DataFrame,
    x: str,
    curves: dict,
    xlabel: str,
    log_x: bool,
    name: str,
    reference: pd.Series | None = None,
    reference_label: str = "",
) -> None:
    """One row per score and one column per corpus and language model. Curves show mean +- std over seeds.

    ``curves`` maps each value of ``long["curve"]`` to (label, color, line style, marker).
    """
    summary = (
        long.groupby(["dataset", "language_model", "measure", "curve", x])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    figure, axes = plt.subplots(len(MEASURES), len(PANELS), figsize=(11, 8.5), squeeze=False)
    for column, (dataset, model) in enumerate(PANELS):
        for row, measure in enumerate(MEASURES):
            axis = axes[row, column]
            panel = summary[
                (summary["dataset"] == dataset) & (summary["language_model"] == model) & (summary["measure"] == measure)
            ]
            for curve, (label, color, style, marker) in curves.items():
                points = panel[panel["curve"] == curve].sort_values(x)
                std = points["std"].fillna(0.0)
                axis.fill_between(
                    points[x], points["mean"] - std, points["mean"] + std, color=color, alpha=0.2, linewidth=0
                )
                axis.plot(
                    points[x],
                    points["mean"],
                    color=color,
                    linestyle=style,
                    marker=marker,
                    markersize=3,
                    linewidth=1,
                    label=label,
                )
            if reference is not None:
                axis.axhline(
                    reference[(dataset, model, measure)],
                    color="#555555",
                    linestyle=":",
                    linewidth=1,
                    label=reference_label,
                )
            if log_x and not panel.empty:  # empty while its runs are missing
                axis.set_xscale("log")
                axis.xaxis.set_major_formatter(ScalarFormatter())
            if row == 0:
                axis.set_title(f"{DATASETS[dataset]}\n{LANGUAGE_MODELS[model]}", fontsize=8)
            if column == 0:
                axis.set_ylabel(MEASURES[measure])
            if row < len(MEASURES) - 1:
                axis.tick_params(labelbottom=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=len(labels), fontsize=7, frameon=True, bbox_to_anchor=(0.5, 1.02)
    )
    figure.supxlabel(xlabel, fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    plotting.save(figure, figures_dir("granularity") / name, summary)


def plot() -> None:
    frame = load_results(results_dir("granularity"))
    clusters = melt(
        frame[frame["analysis"] == "clusters"], r"C=(\d+)/(coarse|random|fine)/(\w+)", ["clusters", "curve", "measure"]
    )
    clusters["clusters"] = clusters["clusters"].astype(int)
    grid(
        clusters,
        "clusters",
        {
            "coarse": ("cluster centers (coarse)", BLUE, "-", "o"),
            "random": ("random points", ORANGE, "--", "s"),
            "fine": ("within a cluster (fine)", GREEN, ":", "^"),
        },
        "number of clusters C",
        log_x=True,
        name="fig22_clusters",
    )

    subspaces = melt(
        frame[frame["analysis"] == "subspaces"], r"p=(\w+)/(principal|random)/(\w+)", ["rank", "curve", "measure"]
    )
    full = subspaces[subspaces["rank"] == "full"].groupby(["dataset", "language_model", "measure"])["value"].mean()
    subspaces = subspaces[subspaces["rank"] != "full"].astype({"rank": int})
    grid(
        subspaces,
        "rank",
        {
            "principal": ("top p principal components", BLUE, "-", "o"),
            "random": ("random p-dimensional subspace", ORANGE, "--", "s"),
        },
        "dimensions p",
        log_x=True,
        name="fig23_subspaces",
        reference=full,
        reference_label="full space",
    )

    clipping = melt(
        frame[frame["analysis"] == "clipping"],
        r"(vision|language)/(below|above)/([+-]\d\.\d)/(\w+)",
        ["space", "side", "threshold", "measure"],
    )
    clipping["threshold"] = clipping["threshold"].astype(float)
    clipping["curve"] = clipping["space"] + ", clipped " + clipping["side"]
    # Clipping from below at -1 leaves the kernel as it is.
    unclipped = clipping[(clipping["curve"] == "vision, clipped below") & (clipping["threshold"] == -1.0)]
    grid(
        clipping,
        "threshold",
        {
            "vision, clipped below": ("vision, clipped below", BLUE, "-", None),
            "language, clipped below": ("language, clipped below", ORANGE, "-", None),
            "vision, clipped above": ("vision, clipped above", BLUE, "--", None),
            "language, clipped above": ("language, clipped above", ORANGE, "--", None),
        },
        "clipping threshold τ on the cosine similarity",
        log_x=False,
        name="fig24_clipping",
        reference=unclipped.groupby(["dataset", "language_model", "measure"])["value"].mean(),
        reference_label="unclipped",
    )


EXPERIMENT = Experiment("granularity", configurations, run, plot, memory_gb=96, hours=6)

if __name__ == "__main__":
    main(EXPERIMENT)
