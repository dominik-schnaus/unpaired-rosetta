"""Faithfulness of the generated images (App. C.2): Table 6 (CyclePrefDB) and Table 7 (MS COCO).

Each prompt is turned into an image by each row of the table (the real photo, Scale-RAE, and the maps of
``experiments/text_to_image.py``) and scored with CLIPScore, VQAScore, TIFA and CycleReward. All maps start from the
same noise for a prompt. Needs one GPU with at least 24 GB and the fits of ``experiments/text_to_image.py``.

    python -m experiments.text_to_image_scores run [--slurm --jobs 43 --gpus 1]
    python -m experiments.text_to_image_scores plot
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments import plotting, text_to_image
from experiments.runner import Experiment, figures_dir, main, results_dir
from experiments.text_to_image import run_in_environment
from modalities.text_to_image import LANGUAGE_MODELS, LANGUAGE_NAMES
from modalities.text_to_image.prompts import NUM_PROMPTS, load_prompts, prompts_path
from unpaired_rosetta.embeddings import storage_root

SHARD_SIZES = {"cycleprefdb": 128, "coco_val2014": 1024}
TABLES = {"cycleprefdb": ("tab6", 3), "coco_val2014": ("tab7", 4)}  # file name and printed decimals
NUM_PAIRS = [0, 1, 10, 100]
METRICS = {"clipscore": "CLIPScore", "vqascore": "VQAScore", "tifa": "TIFA", "cyclereward": "CycleReward"}


@dataclass(frozen=True)
class ScoreShard:
    dataset: str  # "cycleprefdb" or "coco_val2014"
    shard: int

    @property
    def start(
        self,
    ) -> int:
        return self.shard * SHARD_SIZES[self.dataset]

    @property
    def stop(
        self,
    ) -> int:
        return min(self.start + SHARD_SIZES[self.dataset], NUM_PROMPTS[self.dataset])


def aligner_rows() -> dict[str, object]:
    """The table rows that use a fitted map, by name, with their fit from ``experiments/text_to_image.py``."""
    fits = {(c.method, c.model_y, c.num_pairs): c for c in text_to_image.configurations()}
    return {
        f"{method}_{language}_{pairs}": fits[(method, language, pairs)]
        for language in LANGUAGE_MODELS
        for method in text_to_image.METHODS
        for pairs in NUM_PAIRS
    }


def rows() -> list[str]:
    return ["gt", "scale-rae", *aligner_rows()]


def configurations() -> list[ScoreShard]:
    return [
        ScoreShard(dataset, shard)
        for dataset, size in SHARD_SIZES.items()
        for shard in range(math.ceil(NUM_PROMPTS[dataset] / size))
    ]


def work_dir(
    dataset: str,
) -> Path:
    """The images and TIFA questions of a dataset. MS COCO needs about 80 GB."""
    return storage_root() / "samples" / "text_to_image_scores" / dataset


def image_folder(
    dataset: str,
    row: str,
) -> Path:
    return work_dir(dataset) / "images" / row


def run(
    config: ScoreShard,
    num_workers: int,
    num_threads: int,
) -> dict:
    load_prompts(config.dataset)  # written on first use
    prompts = prompts_path(config.dataset)
    shard = ("--prompts", prompts, "--start", config.start, "--stop", config.stop)
    questions = work_dir(config.dataset) / "questions" / f"{config.shard:04d}.json"
    run_in_environment("t2i-metrics", "modalities.text_to_image.tifa", *shard, "--output", questions)
    fits = aligner_rows()
    run_in_environment(
        "text2image",
        "modalities.text_to_image.sample",
        "prompts",
        *shard,
        "--maps",
        *(text_to_image.map_path(fit) for fit in fits.values()),
        "--outputs",
        *(image_folder(config.dataset, row) for row in fits),
    )
    run_in_environment(
        "scale-rae", "modalities.text_to_image.scale_rae", *shard, "--output", image_folder(config.dataset, "scale-rae")
    )
    scores = work_dir(config.dataset) / "scores" / f"{config.shard:04d}.json"
    images = ["gt=prompts"] + [f"{row}={image_folder(config.dataset, row)}" for row in rows()[1:]]
    run_in_environment(
        "t2i-metrics",
        "modalities.text_to_image.scores",
        *shard,
        "--images",
        *images,
        "--questions",
        questions,
        "--output",
        scores,
    )
    return json.loads(scores.read_text())


def per_prompt_scores() -> pd.DataFrame:
    """One line per dataset, row, prompt and metric. TIFA is NaN for prompts without questions."""
    records = []
    for path in sorted(results_dir("text_to_image_scores").glob("*.json")):
        record = json.loads(path.read_text())
        dataset, result = record["config"]["dataset"], record["metrics"]
        for row, metrics in result["scores"].items():
            for metric, values in metrics.items():
                records += [
                    (dataset, row, index, metric, math.nan if value is None else value)
                    for index, value in zip(result["indices"], values)
                ]
    return pd.DataFrame.from_records(records, columns=["dataset", "row", "index", "metric", "score"])


def row_labels(
    row: str,
) -> tuple[str, str, str]:
    """Text encoder, method and pairs of a row, as printed in the tables."""
    if row == "gt":
        return "–", "GT image", "–"
    if row == "scale-rae":
        return "Qwen2.5-1.5B", "Scale-RAE", "–"
    method, language, pairs = row.split("_")
    return LANGUAGE_NAMES[language], method, pairs


def summary(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Mean and standard error over prompts of each row and metric, ignoring NaN."""
    grouped = scores.dropna(subset=["score"]).groupby(["row", "metric"])["score"]
    frame = grouped.agg(mean="mean", std="std", count="count").reset_index()
    frame["standard_error"] = frame["std"] / frame["count"] ** 0.5
    return frame.drop(columns="std")


def table(
    numbers: pd.DataFrame,
    digits: int,
) -> pd.DataFrame:
    lines = []
    for row in rows():
        encoder, method, pairs = row_labels(row)
        line = {"Text encoder": encoder, "Method": method, "Pairs": pairs}
        for metric, label in METRICS.items():
            cell = numbers[(numbers["row"] == row) & (numbers["metric"] == metric)]
            line[label] = (
                plotting.format_mean_std(cell["mean"].item(), cell["standard_error"].item(), digits)
                if len(cell)
                else ""
            )
        lines.append(line)
    return pd.DataFrame(lines)


def plot() -> None:
    scores = per_prompt_scores()
    for dataset, (name, digits) in TABLES.items():
        subset = scores[scores["dataset"] == dataset]
        if subset.empty:
            print(f"{dataset}: no scores yet")
            continue
        numbers = summary(subset)
        prompts = subset[subset["metric"] == "clipscore"].groupby("row")["index"].nunique()
        complete = (prompts == NUM_PROMPTS[dataset]).all() and set(prompts.index) == set(rows())
        title = f"{name[:3].title()}. {name[3:]}: {dataset}, mean ± standard error over prompts"
        if not complete:
            title += f" (INCOMPLETE: {prompts.min()} of {NUM_PROMPTS[dataset]} prompts)"
        without_questions = subset[(subset["row"] == "gt") & (subset["metric"] == "tifa")]["score"].isna().sum()
        print(f"{dataset}: TIFA leaves out {without_questions} prompts without questions")
        figure = plotting.table_figure(table(numbers, digits), title)
        plotting.save(figure, figures_dir("text_to_image_scores") / f"{name}_{dataset}", numbers)


EXPERIMENT = Experiment("text_to_image_scores", configurations, run, plot, cpus=4, gpus=1, memory_gb=64, hours=24)

if __name__ == "__main__":
    main(EXPERIMENT)
