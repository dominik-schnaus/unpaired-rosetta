"""What unpaired vision-language alignment recovers: Fig. 1 (teaser) and Fig. 6a-c.

Fig. 6a-c shows retrieval examples and the aligned space for a cross-dataset fit, with queries picked as k-means
medoids. Fig. 1 shows the nearest caption of three teaser photos, each run using the median seed of the paper.

    python -m experiments.vision_language_examples run
    python -m experiments.vision_language_examples plot
"""

from __future__ import annotations

import json
import textwrap
from collections import Counter
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from experiments import plotting
from experiments.alignment import AlignmentRun, load_training_data, run_alignment
from experiments.runner import Experiment, figures_dir, main
from modalities.image_captions import QWEN3_GEN, datasets
from unpaired_rosetta.embeddings import embedding_path, load_rows, storage_root
from unpaired_rosetta.linalg import center_and_normalize
from unpaired_rosetta.randomness import Randomness
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

VISION_MODEL = "dinov2_vit-b14@224_mean"
RETRIEVAL_RUN = AlignmentRun(
    "coco_train2014", VISION_MODEL, "mpnet", seed=2197670066, dataset_y="StanfordParagraphCaptioning"
)
TEASER_RUN = AlignmentRun("coco_train2014", VISION_MODEL, QWEN3_GEN, seed=2584932063)
TEASER_PHOTOS = [1213, 56053, 48584]  # MS COCO train rows: sheep in a field, a skateboarder, a boy with a pizza
NUM_QUERY_CLUSTERS, NUM_QUERIES, NUM_RETRIEVED = 12, 2, 3
POINTS_PER_CLASS = 400
CONTENT = {"animal": "animals", "vehicle": "vehicles", "food": "food", "sports": "sports", "furniture": "furniture"}
CLASS_COLORS = {
    "animals": "#A2AD00",
    "vehicles": "#C4071B",
    "food": "#E37222",
    "sports": "#0065BD",
    "furniture": "#69085A",
    "people": "#666666",
}


def checkpoint_path(
    name: str,
) -> Path:
    return storage_root() / "checkpoints" / "vision_language_examples" / f"{name}.pt"


def configurations() -> list[AlignmentRun]:
    return [RETRIEVAL_RUN, TEASER_RUN]


def run(
    config: AlignmentRun,
    num_workers: int,
    num_threads: int,
) -> dict:
    aligner, metrics = run_alignment(config, num_workers, num_threads)
    name = "retrieval" if config == RETRIEVAL_RUN else "teaser"
    path = checkpoint_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean_x": aligner.mean_x, "mean_y": aligner.mean_y, "weight": aligner.weight}, path)
    return metrics


def load_aligner(
    name: str,
) -> WassersteinProcrustes:
    state = torch.load(checkpoint_path(name))
    aligner = WassersteinProcrustes()
    aligner.mean_x, aligner.mean_y, aligner.weight = state["mean_x"], state["mean_y"], state["weight"]
    return aligner


def kmeans_labels(
    samples: torch.Tensor,
    num_clusters: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    with threadpool_limits(1):
        model = KMeans(n_clusters=num_clusters, random_state=seed).fit(samples.numpy())
    return model.cluster_centers_, model.labels_


def medoid_queries(
    embeddings: torch.Tensor,
    num_queries: int,
) -> list[int]:
    """The item closest to the centre of each of the largest k-means clusters, skipping near-duplicates."""
    centers, labels = kmeans_labels(embeddings, NUM_QUERY_CLUSTERS)
    queries = []
    for cluster, _ in Counter(labels.tolist()).most_common():
        members = np.flatnonzero(labels == cluster)
        distances = np.linalg.norm(embeddings[members].numpy() - centers[cluster], axis=1)
        medoid = int(members[distances.argmin()])
        if queries and float((embeddings[queries] @ embeddings[medoid]).max()) > 0.9:
            continue
        queries.append(medoid)
        if len(queries) == num_queries:
            break
    return queries


def thumbnail(
    path: Path,
    size: int = 256,
) -> Image.Image:
    image = Image.open(path).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def show_image(
    axis: Axes,
    image: Image.Image,
) -> None:
    axis.imshow(image)
    axis.axis("off")


def coco_classes(
    image_ids: list[int],
) -> list[str | None]:
    """The one content supercategory an image shows, "people" if it shows only persons, else None."""
    instances = json.loads((datasets.raw_root() / "coco2014" / "annotations" / "instances_val2014.json").read_text())
    supercategory = {category["id"]: category["supercategory"] for category in instances["categories"]}
    present = {}
    for annotation in instances["annotations"]:
        present.setdefault(annotation["image_id"], set()).add(supercategory[annotation["category_id"]])
    classes = []
    for image_id in image_ids:
        content = present.get(image_id, set()) & set(CONTENT)
        if len(content) == 1:
            classes.append(CONTENT[content.pop()])
        elif not content and "person" in present.get(image_id, set()):
            classes.append("people")
        else:
            classes.append(None)
    return classes


def joint_pca(
    points_x: torch.Tensor,
    points_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Both sets on the unit sphere, projected by one PCA fitted on their union."""
    joint = torch.cat(
        [torch.nn.functional.normalize(points_x, dim=1), torch.nn.functional.normalize(points_y, dim=1)]
    ).double()
    mean = joint.mean(dim=0, keepdim=True)
    values, vectors = torch.linalg.eigh((joint - mean).T @ (joint - mean))
    basis = vectors[:, -2:].flip(-1)
    projected = ((joint - mean) @ basis).float()
    explained = (values.flip(0)[:2] / values.clamp(min=0).sum()).tolist()
    return projected[: len(points_x)], projected[len(points_x) :], explained


def figure6() -> None:
    aligner = load_aligner("retrieval")
    coco_val = datasets.coco("val2014")
    images = load_rows(embedding_path("coco_val2014", "vision", VISION_MODEL))
    captions = load_rows(embedding_path("coco_val2014", "language", "mpnet"))
    mapped = aligner.transform(images)
    keys = center_and_normalize(captions, aligner.mean_y)

    figure = plt.figure(figsize=(12, 9))
    grid = figure.add_gridspec(2, 2, hspace=0.25, wspace=0.15)
    text_queries = medoid_queries(keys, NUM_QUERIES)
    panel = grid[0, 0].subgridspec(NUM_QUERIES, NUM_RETRIEVED + 1, wspace=0.05)
    for row, query in enumerate(text_queries):
        axis = figure.add_subplot(panel[row, 0])
        axis.axis("off")
        axis.text(0, 0.5, "\n".join(textwrap.wrap(f"“{coco_val.texts[query][0]}”", 24)), va="center", fontsize=8)
        for column, index in enumerate((mapped @ keys[query]).topk(NUM_RETRIEVED).indices.tolist()):
            show_image(figure.add_subplot(panel[row, column + 1]), thumbnail(coco_val.image_paths[index]))
    figure.text(0.3, 0.49, "(a) Text-to-image retrieval.", ha="center")

    image_queries = medoid_queries(mapped, NUM_QUERIES)
    panel = grid[0, 1].subgridspec(NUM_QUERIES, 2, width_ratios=[1, 2.5], wspace=0.05)
    for row, query in enumerate(image_queries):
        show_image(figure.add_subplot(panel[row, 0]), thumbnail(coco_val.image_paths[query]))
        retrieved = (keys @ mapped[query]).topk(NUM_RETRIEVED).indices.tolist()
        lines = [
            f"{rank + 1}. " + textwrap.shorten(coco_val.texts[index][0], 45) for rank, index in enumerate(retrieved)
        ]
        axis = figure.add_subplot(panel[row, 1])
        axis.axis("off")
        axis.text(0, 0.5, "\n".join(lines), va="center", fontsize=8)
    figure.text(0.72, 0.49, "(b) Image-to-text retrieval.", ha="center")

    axis = figure.add_subplot(grid[1, :])
    image_ids = [int(path.stem.split("_")[-1]) for path in coco_val.image_paths]
    classes = coco_classes(image_ids)
    order = Randomness(RETRIEVAL_RUN.seed).permutation(len(classes)).tolist()
    counts, kept = Counter(), []
    for index in order:  # random class-balanced sample, so class frequencies do not dominate the plot
        if classes[index] is not None and counts[classes[index]] < POINTS_PER_CLASS:
            counts[classes[index]] += 1
            kept.append(index)
    image_2d, text_2d, explained = joint_pca(mapped[kept], keys[kept])
    for name, color in CLASS_COLORS.items():
        mask = torch.tensor([classes[index] == name for index in kept])
        axis.scatter(*image_2d[mask].T, s=5, color=color, alpha=0.5, marker="o")
        axis.scatter(*text_2d[mask].T, s=7, color=color, alpha=0.5, marker="^", facecolors="none")
        axis.annotate(name, image_2d[mask].mean(dim=0).tolist(), fontsize=9, weight="bold", color=color)
    axis.scatter([], [], marker="o", color="gray", label="image")
    axis.scatter([], [], marker="^", color="gray", facecolors="none", label="caption")
    axis.legend(fontsize=8)
    axis.set_xlabel(f"PC 1 ({100 * explained[0]:.0f}% of the variance)")
    axis.set_ylabel(f"PC 2 ({100 * explained[1]:.0f}%)")
    axis.set_title("(c) Shared semantic regions after alignment (0 pairs)", fontsize=10)
    plotting.save(figure, figures_dir("vision_language_examples") / "fig6abc_retrieval_and_space")


def figure1() -> None:
    aligner = load_aligner("teaser")
    randomness = Randomness(TEASER_RUN.seed)
    _, captions_half, _, _ = load_training_data(TEASER_RUN, randomness)  # caption half used for fitting
    randomness = Randomness(TEASER_RUN.seed)
    from unpaired_rosetta.evaluation import disjoint_split

    _, caption_rows = disjoint_split(82783, randomness)
    coco_train = datasets.coco("train2014")
    photos = load_rows(embedding_path("coco_train2014", "vision", VISION_MODEL), torch.tensor(TEASER_PHOTOS))
    keys = center_and_normalize(captions_half, aligner.mean_y)
    nearest = (aligner.transform(photos) @ keys.T).argmax(dim=1).tolist()

    figure, axes = plt.subplots(len(TEASER_PHOTOS), 2, figsize=(7, 6), gridspec_kw={"width_ratios": [1, 2]})
    for row, (photo, caption_index) in enumerate(zip(TEASER_PHOTOS, nearest)):
        show_image(axes[row, 0], thumbnail(coco_train.image_paths[photo]))
        axes[row, 1].axis("off")
        caption = coco_train.texts[int(caption_rows[caption_index])][0]
        axes[row, 1].text(0, 0.5, "\n".join(textwrap.wrap(f"“{caption}”", 40)), va="center", fontsize=10)
    figure.suptitle("Nearest caption of each image, without a single pair (DINOv2 B/14, Qwen3-8B)", fontsize=10)
    plotting.save(figure, figures_dir("vision_language_examples") / "fig1_teaser")


def plot() -> None:
    figure6()
    figure1()


EXPERIMENT = Experiment("vision_language_examples", configurations, run, plot, memory_gb=96, hours=6)

if __name__ == "__main__":
    main(EXPERIMENT)
