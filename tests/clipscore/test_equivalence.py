"""Compare our CLIPScore (``modalities/text_to_image/scores.py``) with the official code
(github.com/jmhessel/clipscore, ``get_clip_score``). The score is ``2.5 max(cos(image, caption), 0)`` of the CLIP
ViT-L/14 embeddings.

The official script makes two choices that the paper does not. It puts "A photo depicts " in front of each caption,
and it runs CLIP in float16 on the GPU. The paper embeds the caption alone in float32, so the tests call the official
functions without the prefix and in float32 on the CPU.

The image preprocessing still differs. The Hugging Face CLIP processor and torchvision both resize the shorter side to
224 pixels (bicubic) and crop the center, but they round the longer side differently. A photograph of 2247 x 1500
pixels is therefore cropped one column apart. On images that both pipelines turn into the same pixels, the scores
agree to float32 precision once the GPU computes the patch embedding in full float32. By default cuDNN uses TF32 for
convolutions, which moves our scores by up to 1e-3. The paper kept the default.

Run ``tests/clipscore/clone.sh`` first. The tests need the ``t2i-metrics`` environment, a GPU and the CyclePrefDB
prompts.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.official_code import official_code
from unpaired_rosetta.embeddings import storage_root

pytest.importorskip(
    "clip", reason="runs in the t2i-metrics environment: pixi run -e t2i-metrics pytest tests/clipscore"
)

OFFICIAL = Path(__file__).parent / "official"
PROMPTS = storage_root() / "data" / "text_to_image" / "cycleprefdb" / "prompts.jsonl"
pytestmark = [
    pytest.mark.official,
    pytest.mark.gpu,
    pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/clipscore/clone.sh"),
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


def official_scores(
    images: list[str],
    captions: list[str],
    prefix: str | None,
) -> np.ndarray:
    """``get_clip_score`` with ViT-L/14 in float32 on the CPU and the given caption prefix (None for no prefix)."""
    with official_code(OFFICIAL):
        import clip
        import clipscore

        model, _ = clip.load("ViT-L/14", device="cpu", jit=False)
        model.eval()
        if prefix is None:
            # The dataset adds a space after the prefix, and the CLIP tokenizer strips leading whitespace.
            clipscore.CLIPCapDataset.__init__.__defaults__ = (" ",)
            assert torch.equal(
                clipscore.CLIPCapDataset(captions)[0]["caption"], clip.tokenize(captions[0], truncate=True).squeeze()
            )
        _, per_caption, _ = clipscore.get_clip_score(model, images, captions, "cpu")
    return per_caption


@pytest.fixture(scope="module")
def scorer() -> Callable[[list[str], list[str]], list[float]]:
    from modalities.text_to_image.scores import clipscore_scorer

    return clipscore_scorer()


@pytest.fixture(scope="module")
def ours(
    pairs: tuple[list[str], list[str]],
    scorer: Callable[[list[str], list[str]], list[float]],
) -> np.ndarray:
    return np.array(scorer(*pairs))


def test_equals_the_official_clipscore_on_the_same_pixels(
    pairs: tuple[list[str], list[str]],
    scorer: Callable[[list[str], list[str]], list[float]],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The scores agree when the official transforms resize and crop the photographs first.

    Both pipelines then see the same pixels.
    """
    with official_code(OFFICIAL):
        import clipscore

        resize_and_crop = clipscore.CLIPImageDataset([])._transform_test(224).transforms[:2]
    folder = tmp_path_factory.mktemp("cropped")
    images, captions = pairs
    cropped = []
    for index, path in enumerate(images):
        image = Image.open(path)
        for transform in resize_and_crop:
            image = transform(image)
        cropped.append(str(folder / f"{index}.png"))
        image.convert("RGB").save(cropped[-1])
    official = official_scores(cropped, captions, prefix=None)
    default = np.abs(np.array(scorer(cropped, captions)) - official)
    torch.backends.cudnn.allow_tf32 = False
    try:
        difference = np.abs(np.array(scorer(cropped, captions)) - official)
    finally:
        torch.backends.cudnn.allow_tf32 = True
    print(
        f"CLIPScore on the same pixels, |ours - official|: max {difference.max():.2e} ({default.max():.2e} with TF32 convolutions)"
    )
    assert difference.max() < 1e-5


def test_equals_the_official_clipscore_without_prefix(
    pairs: tuple[list[str], list[str]],
    ours: np.ndarray,
) -> None:
    official = official_scores(*pairs, prefix=None)
    difference = np.abs(ours - official)
    print(
        f"CLIPScore, |ours - official| (no prefix, float32): max {difference.max():.2e}, median {np.median(difference):.2e}"
    )
    # Measured 2026-09-29: median 2e-4 and max 1.5e-2, the latter on the one photograph cropped one column apart.
    assert np.median(difference) < 1e-3
    assert difference.max() < 2e-2


def test_effect_of_the_official_prefix(
    pairs: tuple[list[str], list[str]],
    ours: np.ndarray,
) -> None:
    """Print how much the official "A photo depicts" prefix would change the scores of the paper. Not an equivalence."""
    with_prefix = official_scores(*pairs, prefix="A photo depicts")
    without_prefix = official_scores(*pairs, prefix=None)
    change = with_prefix - without_prefix
    print(f"CLIPScore, official prefix - no prefix: mean {change.mean():+.4f}, max |change| {np.abs(change).max():.4f}")
    assert np.abs(change).max() > 0
