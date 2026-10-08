"""Captions to images: text embedding -> fitted language-to-image map -> RAE diffusion model (``text2image`` env).

A sentence encoder embeds the caption. A map from ``experiments/text_to_image.py`` moves it into the space of the
RAE image embeddings (``conditioning``). The diffusion model then runs 250 Euler steps of the probability-flow ODE
from Gaussian noise without guidance, and the RAE decoder turns the result into an image. The paper draws the noise
in two ways:

* ``grid``: the eight captions of Figs. 6d and 15-18 share one noise tensor drawn on the GPU from the grid seed, in
  float32 and batches of two. Two grids then differ only in the map.
* ``prompts``: each benchmark prompt of Tables 6 and 7 has its own noise seed (``prompt_seed``). All maps then start
  from the same noise for a prompt, whatever the batch. Uses TF32 matmuls and batches of 32.

    pixi run -e text2image python -m modalities.text_to_image.sample grid --maps a.pt b.pt --outputs dir_a dir_b
    pixi run -e text2image python -m modalities.text_to_image.sample prompts --prompts prompts.jsonl --start 0 --stop 128 --maps ... --outputs ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn.functional import normalize

from modalities.text_to_image import (
    CAPTIONS,
    LANGUAGE_MODELS,
    LATENT_SIZE,
    NUM_STEPS,
    RAE_CONFIG,
    RAE_ROOT,
    diffusion_checkpoint,
    prompt_seed,
)
from unpaired_rosetta.linalg import center_and_normalize

if TYPE_CHECKING:
    from PIL import Image
    from torch import nn

GRID_SEED = 734796314  # Seed of the aligner fits in the text-to-image experiments.
GRID_BATCH_SIZE = 2
PROMPT_BATCH_SIZE = 32
TEXT_BATCH_SIZE = 32


def load_diffusion(
    device: torch.device,
) -> tuple[nn.Module, nn.Module, Callable[..., Tensor]]:
    """Return the frozen RAE, the diffusion transformer with the EMA weights of the paper's checkpoint, and the ODE
    sampler with 250 steps."""
    from omegaconf import OmegaConf

    sys.path.insert(0, str(RAE_ROOT / "src"))
    from stage2.transport import Sampler, create_transport
    from utils.model_utils import instantiate_from_config

    config = OmegaConf.load(RAE_CONFIG)
    autoencoder_config = OmegaConf.to_container(config.stage_1, resolve=True)
    for key in ("decoder_config_path", "pretrained_decoder_path", "normalization_stat_path"):
        autoencoder_config["params"][key] = str(RAE_ROOT / autoencoder_config["params"][key])  # Relative to RAE.
    autoencoder = instantiate_from_config(autoencoder_config).eval().to(device).requires_grad_(False)
    model = instantiate_from_config(OmegaConf.to_container(config.stage_2, resolve=True))
    model.load_state_dict(torch.load(diffusion_checkpoint(), map_location="cpu", weights_only=False)["ema"])
    model = model.eval().to(device).requires_grad_(False)

    time_dist_shift = math.sqrt(config.misc.time_dist_shift_dim / config.misc.time_dist_shift_base)
    transport = create_transport(**OmegaConf.to_container(config.transport.params), time_dist_shift=time_dist_shift)
    sampler_parameters = OmegaConf.to_container(config.sampler.params) | {"num_steps": NUM_STEPS}
    sample_fn = Sampler(transport).sample_ode(**sampler_parameters)
    return autoencoder, model, sample_fn


def caption_embeddings(
    language: str,
    captions: list[str],
    device: torch.device,
) -> Tensor:
    """Unit-length sentence embeddings, computed like the stored training embeddings."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(LANGUAGE_MODELS[language], device=str(device))
    return normalize(torch.as_tensor(encoder.encode(list(captions), batch_size=TEXT_BATCH_SIZE)), dim=-1)


def conditioning(
    text: Tensor,
    language_map: dict,
    device: torch.device,
) -> Tensor:
    """Map caption embeddings to image embeddings.

    Centers with the text mean of the fit, maps, adds the image mean back and normalizes like the RAE training
    embeddings.
    """
    mapped = center_and_normalize(text.to(device), language_map["mean_text"].to(device)) @ language_map[
        "text_to_image"
    ].to(device)
    return normalize(mapped + language_map["mean_image"].to(device), dim=-1)


@torch.inference_mode()
def generate(
    autoencoder: nn.Module,
    model: nn.Module,
    sample_fn: Callable[..., Tensor],
    conditions: Tensor,
    noise: Tensor,
    batch_size: int,
) -> list[Image.Image]:
    """One image per condition, with guidance scale 1.0."""
    from torchvision.transforms.functional import to_pil_image

    images = []
    for start in range(0, len(conditions), batch_size):
        latents = sample_fn(noise[start : start + batch_size], model.forward, y=conditions[start : start + batch_size])[
            -1
        ]
        images.extend(to_pil_image(image) for image in autoencoder.decode(latents).clamp(0, 1).cpu())
    return images


def save_image(
    image: Image.Image,
    path: Path,
) -> None:
    """Save under a temporary name first, so a killed job never leaves a truncated image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def sample_grid(
    map_paths: list[Path],
    outputs: list[Path],
    device: torch.device,
) -> None:
    """Write ``<output>/image_<i>.png`` for the eight captions and each map."""
    autoencoder, model, sample_fn = load_diffusion(device)
    maps = [torch.load(path) for path in map_paths]
    texts = {language: caption_embeddings(language, CAPTIONS, device) for language in {m["language"] for m in maps}}
    for language_map, output in zip(maps, outputs):
        generator = torch.Generator(device=device).manual_seed(GRID_SEED)
        noise = torch.randn(len(CAPTIONS), *LATENT_SIZE, device=device, generator=generator)
        conditions = conditioning(texts[language_map["language"]], language_map, device)
        for index, image in enumerate(generate(autoencoder, model, sample_fn, conditions, noise, GRID_BATCH_SIZE)):
            save_image(image, output / f"image_{index}.png")


def sample_prompts(
    prompts: list[dict],
    map_paths: list[Path],
    outputs: list[Path],
    device: torch.device,
) -> None:
    """Write ``<output>/<index>.png`` for each prompt and map. Existing images are skipped."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    autoencoder, model, sample_fn = load_diffusion(device)
    maps = [torch.load(path) for path in map_paths]
    captions = [prompt["prompt"] for prompt in prompts]
    texts = {language: caption_embeddings(language, captions, device) for language in {m["language"] for m in maps}}
    for language_map, output in zip(maps, outputs):
        conditions = conditioning(texts[language_map["language"]], language_map, device)
        missing = [row for row, prompt in enumerate(prompts) if not image_path(output, prompt["index"]).exists()]
        for start in range(0, len(missing), PROMPT_BATCH_SIZE):
            rows = missing[start : start + PROMPT_BATCH_SIZE]
            noise = torch.stack([prompt_noise(prompts[row]["index"]) for row in rows]).to(device)
            images = generate(autoencoder, model, sample_fn, conditions[rows], noise, len(rows))
            for row, image in zip(rows, images):
                save_image(image, image_path(output, prompts[row]["index"]))
            print(f"{output.name}: {start + len(rows)}/{len(missing)} images", flush=True)


def prompt_noise(
    index: int,
) -> Tensor:
    return torch.randn(LATENT_SIZE, generator=torch.Generator().manual_seed(prompt_seed(index)))


def image_path(
    folder: Path,
    index: int,
) -> Path:
    return folder / f"{index:06d}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["grid", "prompts"])
    parser.add_argument("--maps", type=Path, nargs="+", required=True, help="language-to-image maps (.pt)")
    parser.add_argument("--outputs", type=Path, nargs="+", required=True, help="one image folder per map")
    parser.add_argument("--prompts", type=Path, help="prompts.jsonl of modalities.text_to_image.prompts")
    parser.add_argument("--start", type=int, default=0, help="first prompt (line) to sample")
    parser.add_argument("--stop", type=int, default=None, help="one past the last prompt to sample")
    arguments = parser.parse_args()
    device = torch.device("cuda")
    if arguments.mode == "grid":
        sample_grid(arguments.maps, arguments.outputs, device)
    else:
        prompts = [json.loads(line) for line in arguments.prompts.read_text().splitlines()][
            arguments.start : arguments.stop
        ]
        sample_prompts(prompts, arguments.maps, arguments.outputs, device)


if __name__ == "__main__":
    main()
