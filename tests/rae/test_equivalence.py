"""Compare our patch of RAE (``external/patches/rae_embedguidance.diff``) with the official, unpatched RAE code.

The patch adds conditioning on a continuous embedding (``guidance_dim``) next to the class labels of the official
diffusion transformer. The tests check that the submodule carries exactly the patch, that the class-label model is
unchanged (same parameters, same outputs), and what the embedding conditioning computes.

Run ``tests/rae/clone.sh`` first. The tests run in the ``text2image`` environment:
``pixi run -e text2image pytest tests/rae``.
"""

import subprocess
from pathlib import Path

import pytest
import torch

from tests.official_code import official_code

REPOSITORY = Path(__file__).resolve().parents[2]

pytest.importorskip("torchdiffeq", reason="runs in the text2image environment: pixi run -e text2image pytest tests/rae")
OFFICIAL = Path(__file__).parent / "official"
PATCHED = REPOSITORY / "external" / "RAE"
pytestmark = [pytest.mark.official, pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/rae/clone.sh")]

SMALL = dict(
    input_size=4,
    patch_size=1,
    in_channels=8,
    hidden_size=[32, 64],
    depth=[2, 1],
    num_heads=[2, 2],
    class_dropout_prob=0.1,
    num_classes=10,
)


def build(
    source: Path,
    seed: int = 0,
    **options: object,
) -> torch.nn.Module:
    with official_code(source / "src"):
        from stage2.models.DDT import DiTwDDTHead

        torch.manual_seed(seed)
        return DiTwDDTHead(**SMALL, **options).eval()


def inputs(
    seed: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Generator]:
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(3, SMALL["in_channels"], SMALL["input_size"], SMALL["input_size"], generator=generator)
    times = torch.rand(3, generator=generator)
    return latents, times, generator


def test_submodule_carries_exactly_the_patch() -> None:
    applied = subprocess.run(["git", "-C", str(PATCHED), "diff"], capture_output=True, text=True, check=True).stdout
    assert applied == (REPOSITORY / "external" / "patches" / "rae_embedguidance.diff").read_text()


def test_class_label_model_is_unchanged() -> None:
    official, patched = build(OFFICIAL), build(PATCHED)
    official_state, patched_state = official.state_dict(), patched.state_dict()
    assert official_state.keys() == patched_state.keys()
    assert all(torch.equal(official_state[key], patched_state[key]) for key in official_state)
    latents, times, generator = inputs()
    labels = torch.randint(0, SMALL["num_classes"], (3,), generator=generator)
    with torch.no_grad():
        assert torch.equal(official(latents, times, labels), patched(latents, times, labels))


def test_embedding_conditioning() -> None:
    """A two-layer projection of the embedding replaces the class-label table.

    Dropping the condition (for classifier-free guidance) puts a learned null embedding in its place.
    """
    model = build(PATCHED, guidance_dim=6)
    embedder = model.y_embedder
    latents, times, generator = inputs()
    embeddings = torch.randn(3, 6, generator=generator)
    with torch.no_grad():
        assert torch.equal(embedder(embeddings, train=False), embedder.projection(embeddings))
        dropped = embedder(embeddings, train=False, force_drop_ids=torch.tensor([1, 0, 1]))
        substituted = torch.stack([embedder.null_embedding, embeddings[1], embedder.null_embedding])
        assert torch.equal(dropped, embedder.projection(substituted))
        assert model(latents, times, embeddings).shape == latents.shape
