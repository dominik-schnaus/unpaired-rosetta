"""Text-to-image generation through the alignment (Sec. 4.5): Fig. 6d and Figs. 15-18.

Fits our aligner and a linear map between RAE image embeddings and a text encoder with 0 to 100 known pairs, then
generates images for eight fixed captions through each map. All grids use one seed, so two grids differ only in the
map.

    python -m experiments.text_to_image run [--slurm --jobs 32 --workers 4]   # the 32 fits (CPU)
    python -m experiments.text_to_image plot    # samples missing grid images (text2image env, GPU), draws grids
"""

from __future__ import annotations

import os
import subprocess
import textwrap
import time
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from experiments import plotting
from experiments.alignment import AlignmentRun, evaluate, load_training_data
from experiments.methods import build_aligner
from experiments.runner import Experiment, figures_dir, main, results_dir
from modalities.text_to_image import (
    CAPTIONS,
    LANGUAGE_MODELS,
    LANGUAGE_NAMES,
    REPOSITORY,
    TRAINING_DATASET,
    VISION_MODEL,
)
from unpaired_rosetta.randomness import SEEDS, Randomness
from unpaired_rosetta.results import result_path, run_id

if TYPE_CHECKING:
    from unpaired_rosetta.baselines.linear import LinearMap
    from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

SEED = SEEDS[0]
HOLDOUT = 8192
NUM_PAIRS = [0, 1, 2, 5, 10, 20, 50, 100]
METHODS = ["ours", "linear"]
FIGURES = {
    ("ours", "mpnet"): "fig15",
    ("linear", "mpnet"): "fig16",
    ("ours", "contriever"): "fig17",
    ("linear", "contriever"): "fig18",
}
COMPACT_PAIRS, COMPACT_CAPTIONS = [0, 5, 10], 3  # Fig. 6d


def configurations() -> list[AlignmentRun]:
    return [
        AlignmentRun(
            TRAINING_DATASET,
            VISION_MODEL,
            language,
            SEED,
            method,
            num_pairs=num_pairs,
            validation=TRAINING_DATASET,
            holdout=HOLDOUT,
        )
        for language in LANGUAGE_MODELS
        for method in METHODS
        for num_pairs in NUM_PAIRS
    ]


def map_path(
    config: AlignmentRun,
) -> Path:
    """The language-to-image map of a fit, next to its result."""
    return results_dir("text_to_image") / "maps" / f"{run_id(config)}.pt"


def language_to_image_map(
    aligner: WassersteinProcrustes | LinearMap,
    config: AlignmentRun,
    randomness: Randomness,
) -> dict:
    """What the sampler needs to map a text embedding into the image space.

    The sampler computes ``normalize(center_and_normalize(t, mean_text) @ text_to_image + mean_image)``. Our map is
    semi-orthogonal, so its transpose inverts it. The linear map is inverted by its pseudo-inverse.
    """
    if config.method == "ours":
        text_to_image = aligner.weight.T
    elif aligner.weight is None:  # linear map without pairs, so use a random map
        text_to_image = torch.linalg.pinv(
            torch.randn(aligner.mean_x.shape[1], aligner.mean_y.shape[1], generator=randomness.torch)
        )
    else:
        text_to_image = torch.linalg.pinv(aligner.weight)
    return {
        "language": config.model_y,
        "mean_image": aligner.mean_x,
        "mean_text": aligner.mean_y,
        "text_to_image": text_to_image,
    }


def run(
    config: AlignmentRun,
    num_workers: int,
    num_threads: int,
) -> dict:
    """Fit, save the language-to-image map, and evaluate on the held-out pairs."""
    start = time.time()
    randomness = Randomness(config.seed)
    samples_x, samples_y, paired_x, paired_y = load_training_data(config, randomness)
    aligner = build_aligner(config.method, num_workers=num_workers, num_threads=num_threads)
    aligner.fit(samples_x, samples_y, paired_x, paired_y, randomness)
    path = map_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(language_to_image_map(aligner, config, randomness), path)
    metrics = evaluate(aligner, config)
    metrics["fit_and_evaluation_seconds"] = time.time() - start
    return metrics


def grid_folder(
    config: AlignmentRun,
) -> Path:
    """The eight images of a fit: ``image_<i>.png`` for caption ``i``."""
    return results_dir("text_to_image") / "grids" / f"{config.method}_{config.model_y}" / str(config.num_pairs)


def sample_missing_grids() -> None:
    """Sample the grid images of each finished fit whose images are missing or older than its map."""
    missing = [
        c for c in configurations() if result_path(results_dir("text_to_image"), c).exists() and grid_is_stale(c)
    ]
    if not missing:
        return
    print(f"sampling the grid images of {len(missing)} fits (text2image environment, one GPU)")
    run_in_environment(
        "text2image",
        "modalities.text_to_image.sample",
        "grid",
        "--maps",
        *map(map_path, missing),
        "--outputs",
        *map(grid_folder, missing),
    )


def grid_is_stale(
    config: AlignmentRun,
) -> bool:
    last_image = grid_folder(config) / f"image_{len(CAPTIONS) - 1}.png"
    return not last_image.exists() or (
        map_path(config).exists() and last_image.stat().st_mtime < map_path(config).stat().st_mtime
    )


def run_in_environment(
    environment: str,
    module: str,
    *arguments: object,
) -> None:
    """Run ``python -m module arguments`` in another pixi environment."""
    command = [
        "pixi",
        "run",
        "--manifest-path",
        str(REPOSITORY / "pyproject.toml"),
        "-e",
        environment,
        "python",
        "-m",
        module,
        *map(str, arguments),
    ]
    variables = {key: value for key, value in os.environ.items() if not key.startswith(("PIXI_", "CONDA_"))}
    subprocess.run(command, cwd=REPOSITORY, env=variables | {"TOKENIZERS_PARALLELISM": "false"}, check=True)


def draw_grid(
    images: dict,
    rows: list[int],
    captions: list[str],
    path: Path,
    transpose: bool = False,
) -> None:
    """Captions as column heads and pair counts as rows (Figs. 15-18), or the transpose (Fig. 6d)."""
    heads = [f"{pairs} pairs" for pairs in rows] if transpose else captions
    sides = captions if transpose else [str(pairs) for pairs in rows]
    figure, axes = plt.subplots(
        len(sides),
        len(heads),
        figsize=(1.2 * len(heads) + 1, 1.2 * len(sides) + 0.6),
        gridspec_kw={"wspace": 0.03, "hspace": 0.03},
        squeeze=False,
    )
    for row, side in enumerate(sides):
        for column, head in enumerate(heads):
            pairs, caption = (rows[column], captions[row]) if transpose else (rows[row], captions[column])
            axis = axes[row, column]
            axis.imshow(Image.open(images[(pairs, caption)]))
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(textwrap.fill(head, 18), fontsize=6)
            if column == 0:
                axis.set_ylabel(textwrap.fill(side, 22) if transpose else side, fontsize=6 if transpose else 8)
    if not transpose:
        figure.supylabel("number of known pairs", fontsize=8)
    numbers = pd.DataFrame(
        [{"num_pairs": pairs, "caption": caption, "image": str(image)} for (pairs, caption), image in images.items()]
    )
    plotting.save(figure, path, numbers)


def plot() -> None:
    sample_missing_grids()
    frame = pd.DataFrame(
        [
            {"method": c.method, "language": c.model_y, "num_pairs": c.num_pairs, "folder": grid_folder(c)}
            for c in configurations()
        ]
    )
    output = figures_dir("text_to_image")
    for (method, language), name in FIGURES.items():
        rows = frame[(frame["method"] == method) & (frame["language"] == language)]
        images = {
            (r.num_pairs, caption): r.folder / f"image_{i}.png"
            for r in rows.itertuples()
            for i, caption in enumerate(CAPTIONS)
        }
        if not all(path.exists() for path in images.values()):
            print(f"skipping {name}: grid images missing")
            continue
        draw_grid(images, NUM_PAIRS, CAPTIONS, output / f"{name}_{method}_{language}")
        if (method, language) == ("ours", "mpnet"):
            compact = {
                key: value
                for key, value in images.items()
                if key[0] in COMPACT_PAIRS and key[1] in CAPTIONS[:COMPACT_CAPTIONS]
            }
            draw_grid(compact, COMPACT_PAIRS, CAPTIONS[:COMPACT_CAPTIONS], output / "fig6d_ours_mpnet", transpose=True)
    fits_table()


def fits_table() -> None:
    """FOSCTTM of each fit on the 8192 held-out pairs. It is not in the paper."""
    from unpaired_rosetta.results import load_results

    results = load_results(results_dir("text_to_image"))
    if results.empty:
        return
    results["language"] = results["model_y"].map(LANGUAGE_NAMES)
    table = results.pivot_table(index="num_pairs", columns=["language", "method"], values="foscttm")
    table.to_csv(figures_dir("text_to_image") / "foscttm_of_the_fits.csv")
    print(table.round(4).to_string())


EXPERIMENT = Experiment("text_to_image", configurations, run, plot, cpus=4, memory_gb=48, hours=12)

if __name__ == "__main__":
    main(EXPERIMENT)
