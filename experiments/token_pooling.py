"""Alignment with generative token pooling (App. D): Fig. 20 (six corpora) and Fig. 21 (caption-detail ladder).

Tests whether generative pooling helps because captions are short. Three poolings of the same captions (generative,
mean and Qwen3-Embedding) are scored by clipped CKA against DINOv2 image embeddings, and their differences are plotted
against the mean caption length. Counting caption lengths needs the raw captions.

    python -m experiments.token_pooling run
    python -m experiments.token_pooling plot
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import Tensor

from experiments import plotting
from experiments.runner import Experiment, figures_dir, main, results_dir
from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.embeddings import embedding_path, reduce_texts
from unpaired_rosetta.geometry import clipped_cka
from unpaired_rosetta.results import load_results

if TYPE_CHECKING:
    from matplotlib.axes import Axes

SLICE_SIZE, NUM_SLICES = 1024, 5
VISION = "dinov2_vit-b14@224_mean"
CORPORA = {  # sorted by mean caption length
    "coco_train2014": "MS COCO",
    "cc12m": "CC12M",
    "wit_train": "WIT",
    "StanfordParagraphCaptioning": "SPC",
    "DOCCIDataset": "DOCCI",
    "DenselyCaptionedImages": "DCI",
}
WEB_CORPORA = {"cc12m", "wit_train"}  # only their first 5 x 1024 items were embedded
# pooling -> language model, for the corpora (Fig. 20) and the ladder (Fig. 21)
POOLINGS = {
    "generative": "qwen3-8b-gen_69445574b3021b8a2733e87d8603d769",
    "mean": "qwen3-8b-mean",
    "embedding": "qwen3-embedding-8b",
}
LADDER_POOLINGS = {
    "generative": "qwen3-1.7b-gen_1aa1557c74992c52c60f81e5a1bd6621",
    "mean": "qwen3-1.7b-mean",
    "embedding": "qwen3-embedding-0.6b",
}
TOKENIZERS = {"corpus": "Qwen/Qwen3-8B", "ladder": "Qwen/Qwen3-1.7B"}  # backbone of generative and mean pooling
# Ladder rungs, stored in ``<corpus>_ladder-<rung>`` folders. "full-keywords_6" is the full caption reduced to its
# 6 rarest content words.
LADDER = {
    "DOCCIDataset": ["full", "half", "first_sentence", "keywords_12", "keywords_6", "keywords_3", "keywords_6_ordered"],
    "DenselyCaptionedImages": ["full", "extended", "short", "full-keywords_12", "full-keywords_6", "full-keywords_3"],
}
CONTRASTS = {"mean": "generative - mean", "embedding": "generative - embedding"}
NUM_THREADS = 3  # as in the paper's runs, since matrix products change in the last bits with the thread count


@dataclass(frozen=True)
class PoolingRun:
    corpus: str
    captions: str  # "corpus" for the corpus' own captions (Fig. 20), else a ladder rung (Fig. 21)
    pooling: str
    language_model: str
    slice: int
    vision_model: str = VISION


@dataclass(frozen=True)
class CaptionLengthRun:
    corpus: str
    captions: str  # as in PoolingRun


def configurations() -> list:
    settings = [(corpus, "corpus", POOLINGS) for corpus in CORPORA]
    settings += [(corpus, rung, LADDER_POOLINGS) for corpus, rungs in LADDER.items() for rung in rungs]
    runs = [CaptionLengthRun(corpus, captions) for corpus, captions, _ in settings]
    runs += [
        PoolingRun(corpus, captions, pooling, model, index)
        for corpus, captions, models in settings
        for pooling, model in models.items()
        for index in range(NUM_SLICES)
    ]
    return runs


def text_folder(
    corpus: str,
    captions: str,
) -> str:
    """The dataset folder of the captions: the corpus' first 5 x 1024 items, or one ladder rung."""
    return f"{corpus}_tokenpooling" if captions == "corpus" else f"{corpus}_ladder-{captions}"


def language_path(
    config: PoolingRun,
) -> Path:
    """The full corpus embedding if one exists for the model, else the embedding of the first 5 x 1024 items."""
    if config.captions == "corpus" and config.corpus not in WEB_CORPORA and config.pooling != "mean":
        return embedding_path(config.corpus, "language", config.language_model)
    return embedding_path(text_folder(config.corpus, config.captions), "language", config.language_model)


def load_slice(
    path: Path,
    index: int,
) -> Tensor:
    """Slice ``index`` of the first 5 x 1024 items as unit-length float32 embeddings."""
    stored = torch.load(path, mmap=True)[index * SLICE_SIZE : (index + 1) * SLICE_SIZE]
    return reduce_texts(stored.float())


def caption_length(
    config: CaptionLengthRun,
) -> dict:
    """Mean number of Qwen3 tokens per caption. Every caption of an item counts."""
    from transformers import AutoTokenizer

    from modalities.image_captions.datasets import texts

    captions = [caption for item in texts(text_folder(config.corpus, config.captions)) for caption in item]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZERS["corpus" if config.captions == "corpus" else "ladder"])
    lengths = torch.tensor(
        [len(ids) for ids in tokenizer(captions, add_special_tokens=False)["input_ids"]], dtype=torch.float
    )
    return {"tokens_mean": lengths.mean().item(), "num_captions": len(captions)}


def run(
    config: PoolingRun | CaptionLengthRun,
    num_workers: int = 1,
    num_threads: int = 1,
) -> dict:
    if isinstance(config, CaptionLengthRun):
        return caption_length(config)
    with torch_threads(NUM_THREADS):
        vision = load_slice(embedding_path(config.corpus, "vision", config.vision_model), config.slice)
        language = load_slice(language_path(config), config.slice)
        return {"clipped_cka": clipped_cka(vision, language)}


STYLES = {  # contrast -> (color, line style, marker)
    "generative - mean": (plotting.PALETTE[0], "-", "D"),
    "generative - embedding": (plotting.PALETTE[1], "--", "s"),
}


def contrasts(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Generative minus each other pooling, per setting and slice, with the mean caption length."""
    alignment = frame[frame["pooling"].notna()]
    lengths = frame[frame["tokens_mean"].notna()][["corpus", "captions", "tokens_mean"]]
    wide = alignment.pivot_table(index=["corpus", "captions", "slice"], columns="pooling", values="clipped_cka")
    parts = [
        pd.DataFrame({"contrast": label, "delta": wide["generative"] - wide[reference]}).reset_index()
        for reference, label in CONTRASTS.items()
    ]
    return pd.concat(parts, ignore_index=True).merge(lengths, on=["corpus", "captions"])


def tick_label(
    length: float,
) -> str:
    return f"{length:.1f}" if length < 10 else f"{length:.0f}"


def contrast_panel(
    axis: Axes,
    deltas: pd.DataFrame,
    names: dict[float, str] | None = None,
) -> pd.DataFrame:
    """Mean +- std over slices of both differences against the mean caption length on a log scale."""
    summary = deltas.groupby(["contrast", "tokens_mean"])["delta"].agg(["mean", "std"]).reset_index()
    axis.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":")
    for contrast, (color, style, marker) in STYLES.items():
        points = summary[summary["contrast"] == contrast].sort_values("tokens_mean")
        axis.fill_between(
            points["tokens_mean"],
            points["mean"] - points["std"],
            points["mean"] + points["std"],
            color=color,
            alpha=0.2,
            linewidth=0,
        )
        axis.plot(
            points["tokens_mean"],
            points["mean"],
            color=color,
            linestyle=style,
            marker=marker,
            markersize=4,
            linewidth=1.2,
            label=contrast,
        )
    lengths = sorted(summary["tokens_mean"].unique())
    axis.set_xscale("log")
    axis.set_xticks(
        lengths, [tick_label(length) + (f"\n({names[length]})" if names else "") for length in lengths], fontsize=7
    )
    axis.minorticks_off()
    axis.set_xlabel("mean caption length (tokens)")
    axis.set_ylabel("Δ CKA")
    axis.legend(fontsize=7)
    return summary


def plot() -> None:
    deltas = contrasts(load_results(results_dir("token_pooling")))

    corpora = deltas[deltas["captions"] == "corpus"]
    names = {length: CORPORA[corpus] for corpus, length in corpora.groupby("corpus")["tokens_mean"].first().items()}
    figure, axis = plt.subplots(figsize=(7.5, 2.6))
    summary = contrast_panel(axis, corpora, names)
    plotting.save(
        figure,
        figures_dir("token_pooling") / "fig20_token_pooling",
        summary.assign(corpus=summary["tokens_mean"].map(names)),
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    summaries = []
    for axis, corpus in zip(axes, LADDER):
        rungs = deltas[(deltas["corpus"] == corpus) & (deltas["captions"] != "corpus")]
        summary = contrast_panel(axis, rungs)
        axis.set_title(CORPORA[corpus], fontsize=9)
        rung_of = rungs.groupby("tokens_mean")["captions"].first()
        summaries.append(summary.assign(corpus=corpus, rung=summary["tokens_mean"].map(rung_of)))
    figure.tight_layout()
    plotting.save(
        figure, figures_dir("token_pooling") / "fig21_caption_ladder", pd.concat(summaries, ignore_index=True)
    )
    print(corpora.groupby(["corpus", "contrast"])["delta"].mean().unstack().round(3).to_string())


EXPERIMENT = Experiment("token_pooling", configurations, run, plot, memory_gb=16, hours=2)

if __name__ == "__main__":
    main(EXPERIMENT)
