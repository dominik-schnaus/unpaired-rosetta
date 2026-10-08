"""Compare our VQAScore (``modalities/text_to_image/scores.py``) with the official t2v_metrics code
(github.com/linzhiqiu/t2v_metrics at its pip release 1.1). The score is the probability that CLIP-FlanT5-XL
answers "Yes".

The official ``VQAScore(images, texts)`` scores each image against each text, one image with all texts at a time.
We score paired lists in batches of 32 pairs through the ``forward`` of the model. The model runs in bfloat16, where
the shape of a batch changes the result by up to a few 1e-2.

Run ``tests/vqascore/clone.sh`` first. The test needs the ``t2i-metrics`` environment, a GPU and the CyclePrefDB
prompts.
"""

import gc
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.official_code import official_code
from unpaired_rosetta.embeddings import storage_root

pytest.importorskip(
    "t2v_metrics", reason="runs in the t2i-metrics environment: pixi run -e t2i-metrics pytest tests/vqascore"
)

OFFICIAL = Path(__file__).parent / "official"
PROMPTS = storage_root() / "data" / "text_to_image" / "cycleprefdb" / "prompts.jsonl"
pytestmark = [
    pytest.mark.official,
    pytest.mark.gpu,
    pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/vqascore/clone.sh"),
    pytest.mark.skipif(not PROMPTS.exists(), reason="run python -m modalities.text_to_image.prompts"),
]
NUM_PROMPTS = 16


@pytest.fixture(scope="module")
def pairs() -> tuple[list[str], list[str]]:
    """Photographs with their own prompt and with the prompt of the next photograph, to cover a range of scores."""
    prompts = [json.loads(line) for line in PROMPTS.read_text().splitlines()[:NUM_PROMPTS]]
    images = [prompt["image"] for prompt in prompts] * 2
    captions = [prompt["prompt"] for prompt in prompts] + [
        prompts[(i + 1) % NUM_PROMPTS]["prompt"] for i in range(NUM_PROMPTS)
    ]
    return images, captions


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def test_equals_the_official_vqascore(
    pairs: tuple[list[str], list[str]],
) -> None:
    """One pair at a time, our call equals the official one.

    In batches of 32 (as in the paper), the bfloat16 kernels of a different batch shape move a probability by up to a
    few 1e-2. Padding is not the cause, because a batch of four identical copies of one pair moves it as much.
    """
    from modalities.text_to_image.scores import VQASCORE_MODEL, vqascore_scorer, weights

    images, captions = pairs
    ours = np.array(vqascore_scorer()(images, captions))
    free_gpu()
    with official_code(OFFICIAL):
        import t2v_metrics

        scorer = t2v_metrics.VQAScore(model=VQASCORE_MODEL, device="cuda", cache_dir=weights("t2v_metrics"))
        with torch.inference_mode():
            official = np.array(
                [scorer(images=[image], texts=[caption])[0, 0].item() for image, caption in zip(images, captions)]
            )
            ours_one_at_a_time = np.array(
                [scorer.model.forward([image], [caption]).float().item() for image, caption in zip(images, captions)]
            )
            copies = np.array(
                [
                    scorer.model.forward([image] * 4, [caption] * 4).float()[0].item()
                    for image, caption in zip(images, captions)
                ]
            )
        del scorer
    free_gpu()
    batched, shape_only = np.abs(ours - official), np.abs(copies - official)
    print(
        f"VQAScore, |ours in batches of 32 - official|: max {batched.max():.2e}, median {np.median(batched):.2e}; "
        f"four identical copies: max {shape_only.max():.2e}; scores from {official.min():.3f} to {official.max():.3f}"
    )
    assert np.array_equal(ours_one_at_a_time, official)
    # Measured 2026-09-30 on an RTX 4500 Ada: max 2.5e-2, median 1.7e-3 (identical copies: max 1.8e-2).
    assert batched.max() < 5e-2 and np.median(batched) < 5e-3
    assert np.corrcoef(ours, official)[0, 1] > 0.999
