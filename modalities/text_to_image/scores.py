"""The four scores of Tables 6 and 7. Each compares an image with its prompt (``t2i-metrics`` env).

* CLIPScore (Hessel et al., 2021): ``2.5 max(cos(image, caption), 0)`` of the CLIP ViT-L/14 embeddings, without a
  prompt prefix.
* VQAScore (Lin et al., 2024): the probability that CLIP-FlanT5-XL answers "Yes" to "Does this figure show
  '<caption>'? Please answer yes or no." (t2v-metrics).
* TIFA (Hu et al., 2023): the fraction of the prompt's questions that BLIP-large answers correctly (``tifa.py``).
  NaN for a prompt without questions.
* CycleReward (Bahng et al., 2025), Combo checkpoint: an unnormalized reward, so only differences mean something.

Each scorer loads its model once and returns a scoring function for many lists of images.

    pixi run -e t2i-metrics python -m modalities.text_to_image.scores --prompts prompts.jsonl --start 0 --stop 128 \
        --images gt=<from prompts> ours_mpnet_0=<folder> ... --questions questions.json --output scores.json
"""

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image

from unpaired_rosetta.embeddings import storage_root

CLIP_MODEL = "openai/clip-vit-large-patch14"
VQASCORE_MODEL = "clip-flant5-xl"
CYCLEREWARD_MODEL = "CycleReward-Combo"
METRICS = ("clipscore", "vqascore", "tifa", "cyclereward")


def weights(
    name: str,
) -> str:
    """Weight folder for t2v-metrics and CycleReward. Both default to the working directory otherwise."""
    return str(storage_root() / "weights" / name)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def open_rgb(
    path: str | Path,
) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def clipscore_scorer(
    batch_size: int = 64,
) -> Callable[[list[str | Path], list[str]], list[float]]:
    """CLIPScore. The projections are applied by hand because ``get_text_features`` differs between transformers
    versions."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_MODEL).eval().to(device())
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    @torch.inference_mode()
    def score(
        images: list[str | Path],
        captions: list[str],
    ) -> list[float]:
        scores = []
        for start in range(0, len(images), batch_size):
            tokens = processor(
                text=captions[start : start + batch_size], return_tensors="pt", padding=True, truncation=True
            ).to(device())
            text = torch.nn.functional.normalize(
                model.text_projection(model.text_model(**tokens).pooler_output), dim=-1
            )
            pixels = processor(
                images=[open_rgb(path) for path in images[start : start + batch_size]], return_tensors="pt"
            ).to(device())
            image = torch.nn.functional.normalize(
                model.visual_projection(model.vision_model(**pixels).pooler_output), dim=-1
            )
            scores += (2.5 * (image * text).sum(-1).clamp_min(0.0)).tolist()
        return scores

    return score


def vqascore_scorer(
    batch_size: int = 32,
) -> Callable[[list[str | Path], list[str]], list[float]]:
    """VQAScore on paired lists. The scorer's own ``__call__`` would score every image against every text.

    The model runs in bfloat16, so a batch of 32 moves a probability by up to a few 1e-2 from one-at-a-time scores.
    """
    import t2v_metrics

    scorer = t2v_metrics.VQAScore(model=VQASCORE_MODEL, device=device(), cache_dir=weights("t2v_metrics"))

    @torch.inference_mode()
    def score(
        images: list[str | Path],
        captions: list[str],
    ) -> list[float]:
        scores = []
        for start in range(0, len(images), batch_size):
            batch_images = [str(path) for path in images[start : start + batch_size]]
            scores += scorer.model.forward(batch_images, list(captions[start : start + batch_size])).float().tolist()
        return scores

    return score


def cyclereward_scorer(
    batch_size: int = 32,
) -> Callable[[list[str | Path], list[str]], list[float]]:
    """CycleReward in batches of 32, as for the paper.

    In a batch, cuDNN runs the patch embedding in TF32 (PyTorch's default for convolutions). A reward then moves by up
    to 2e-2 from one-at-a-time scores (median 3e-4).
    """
    from cyclereward import cyclereward

    model, preprocess = cyclereward(device=device(), model_type=CYCLEREWARD_MODEL, cache_dir=weights("cyclereward"))

    @torch.inference_mode()
    def score(
        images: list[str | Path],
        captions: list[str],
    ) -> list[float]:
        scores = []
        for start in range(0, len(images), batch_size):
            pixels = torch.stack([preprocess(open_rgb(path)) for path in images[start : start + batch_size]]).to(
                device()
            )
            scores += model.score(pixels, list(captions[start : start + batch_size])).float().flatten().tolist()
        return scores

    return score


def tifa_scorer(
    batch_size: int = 16,
) -> Callable[[list[str | Path], list[str], list[list[dict]]], list[float]]:
    from modalities.text_to_image.tifa import tifa_scorer as load

    return load(batch_size)


SCORERS = {
    "clipscore": clipscore_scorer,
    "vqascore": vqascore_scorer,
    "tifa": tifa_scorer,
    "cyclereward": cyclereward_scorer,
}


def score_rows(
    prompts: list[dict],
    rows: dict[str, list],
    questions: dict[str, list] | None,
) -> dict:
    """``{row: {metric: [score of each prompt]}}``. A TIFA score without questions is ``None``."""
    captions = [prompt["prompt"] for prompt in prompts]
    scores = {row: {} for row in rows}
    for metric in METRICS:
        scorer = SCORERS[metric]()
        for row, images in rows.items():
            if metric == "tifa":
                values = scorer(images, captions, [questions[str(prompt["index"])] for prompt in prompts])
            else:
                values = scorer(images, captions)
            scores[row][metric] = [None if math.isnan(value) else value for value in values]
            print(f"{metric} {row}: {len(values)} images", flush=True)
        del scorer
        torch.cuda.empty_cache()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--images", nargs="+", required=True, help="row=folder (images <index>.png), or gt=prompts")
    parser.add_argument("--questions", type=Path, required=True, help="the TIFA questions of these prompts")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    prompts = [json.loads(line) for line in arguments.prompts.read_text().splitlines()][
        arguments.start : arguments.stop
    ]
    rows = {}
    for item in arguments.images:
        row, folder = item.split("=", 1)
        rows[row] = [
            prompt["image"] if folder == "prompts" else f"{folder}/{prompt['index']:06d}.png" for prompt in prompts
        ]
    questions = json.loads(arguments.questions.read_text())["questions"]
    scores = score_rows(prompts, rows, questions)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps({"indices": [prompt["index"] for prompt in prompts], "scores": scores}))


if __name__ == "__main__":
    main()
