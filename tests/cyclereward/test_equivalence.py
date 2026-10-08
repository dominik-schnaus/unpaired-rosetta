"""Compare our CycleReward (``modalities/text_to_image/scores.py``) with the official code
(github.com/hjbahng/cyclereward, ``CycleReward.score``) with the Combo checkpoint. The official package directory
equals the pinned pip release 0.1.7.

The official example scores one image at a time, and we score batches of 32 (as for the paper). Both pad each caption
to the maximum length of the model, so batching changes no input. Batching does change the convolution algorithm of
the patch embedding in the image encoder. For batches, cuDNN picks a TF32 kernel (PyTorch allows TF32 convolutions by
default), which moves a reward by up to 2e-2 (median 3e-4). Without TF32 convolutions, batched and official agree to
1e-5.

Run ``tests/cyclereward/clone.sh`` first. The test needs the ``t2i-metrics`` environment, a GPU and the CyclePrefDB
prompts.
"""

import gc
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.official_code import official_code
from unpaired_rosetta.embeddings import storage_root

pytest.importorskip(
    "cyclereward", reason="runs in the t2i-metrics environment: pixi run -e t2i-metrics pytest tests/cyclereward"
)

OFFICIAL = Path(__file__).parent / "official"
PROMPTS = storage_root() / "data" / "text_to_image" / "cycleprefdb" / "prompts.jsonl"
pytestmark = [
    pytest.mark.official,
    pytest.mark.gpu,
    pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/cyclereward/clone.sh"),
    pytest.mark.skipif(not PROMPTS.exists(), reason="run python -m modalities.text_to_image.prompts"),
]
NUM_PROMPTS = 16


@pytest.fixture(scope="module")
def pairs() -> tuple[list[str], list[str]]:
    """Photographs with their own prompt and with the prompt of the next photograph, to cover a range of rewards."""
    prompts = [json.loads(line) for line in PROMPTS.read_text().splitlines()[:NUM_PROMPTS]]
    images = [prompt["image"] for prompt in prompts] * 2
    captions = [prompt["prompt"] for prompt in prompts] + [
        prompts[(i + 1) % NUM_PROMPTS]["prompt"] for i in range(NUM_PROMPTS)
    ]
    return images, captions


def test_equals_the_official_cyclereward(
    pairs: tuple[list[str], list[str]],
) -> None:
    from modalities.text_to_image.scores import CYCLEREWARD_MODEL, cyclereward_scorer, weights

    images, captions = pairs
    ours = np.array(cyclereward_scorer()(images, captions))
    gc.collect()
    torch.cuda.empty_cache()
    with official_code(OFFICIAL):
        from cyclereward import cyclereward

        model, preprocess = cyclereward(device="cuda", model_type=CYCLEREWARD_MODEL, cache_dir=weights("cyclereward"))
        with torch.inference_mode():
            pixels = [preprocess(Image.open(path).convert("RGB")).unsqueeze(0).cuda() for path in images]
            official = np.array([model.score(image, caption).item() for image, caption in zip(pixels, captions)])
    torch.backends.cudnn.allow_tf32 = False
    try:
        ours_without_tf32 = np.array(cyclereward_scorer()(images, captions))
    finally:
        torch.backends.cudnn.allow_tf32 = True
    batched, exact = np.abs(ours - official), np.abs(ours_without_tf32 - official)
    print(
        f"CycleReward, |ours in batches of 32 - official one at a time|: max {batched.max():.2e}, median {np.median(batched):.2e}; "
        f"without TF32 convolutions: max {exact.max():.2e}; rewards from {official.min():.2f} to {official.max():.2f}"
    )
    # Measured 2026-09-30 on an RTX 4500 Ada: max 1.7e-2 and median 2.7e-4, and 5e-6 without TF32 convolutions.
    assert exact.max() < 1e-4
    assert batched.max() < 5e-2
