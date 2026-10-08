"""Embed the cached DESI galaxies in float32. Images use AstroDINO (CLS token). Spectra use SpecFormer (mean of the
last hidden states over positions).

AstroDINO reads 144 x 144 center crops of the stretched RGB images. AstroCLIP's data loader permutes
``[B, H, W, C]`` to ``[B, C, W, H]``, so height and width are swapped here too, as in training. SpecFormer reads the
flux as a ``[B, 7781, 1]`` sequence and standardizes and slices it itself.

    pixi run python -m modalities.galaxy_spectra.embed
"""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from modalities.galaxy_spectra import CACHE, CHECKPOINTS, IMAGE, SPECTRUM
from modalities.galaxy_spectra.astrodino import load_astrodino
from modalities.galaxy_spectra.specformer import load_specformer

CROP = 144


def image_batch(
    images: np.ndarray,
) -> Tensor:
    """Turn ``[B, H, W, 3]`` RGB images into ``[B, 3, 144, 144]`` crops with height and width swapped."""
    batch = torch.from_numpy(np.asarray(images, dtype=np.float32)).permute(0, 3, 2, 1)
    top, left = (batch.shape[2] - CROP) // 2, (batch.shape[3] - CROP) // 2
    return batch[:, :, top : top + CROP, left : left + CROP]


@torch.inference_mode()
def embed_images(
    model: nn.Module,
    images: np.ndarray,
    batch_size: int = 128,
    device: str = "cuda",
) -> Tensor:
    batches = [
        model(image_batch(images[start : start + batch_size]).to(device)).cpu()
        for start in range(0, len(images), batch_size)
    ]
    return torch.cat(batches)


@torch.inference_mode()
def embed_spectra(
    model: nn.Module,
    spectra: np.ndarray,
    batch_size: int = 64,
    device: str = "cuda",
) -> Tensor:
    batches = []
    for start in range(0, len(spectra), batch_size):
        batch = torch.from_numpy(np.array(spectra[start : start + batch_size], dtype=np.float32))[:, :, None]
        batches.append(model(batch.to(device))["embedding"].mean(dim=1).cpu())
    return torch.cat(batches)


def save(
    tensor: Tensor,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    print(f"wrote {path} {tuple(tensor.shape)}")


def main(
    device: str = "cuda",
) -> None:
    if not IMAGE.exists():
        model = load_astrodino(CHECKPOINTS / "astrodino.ckpt").to(device)
        save(embed_images(model, np.load(CACHE / "images.npy", mmap_mode="r"), device=device), IMAGE.path)
    if not SPECTRUM.exists():
        model = load_specformer(CHECKPOINTS / "specformer.ckpt").eval().to(device)
        save(embed_spectra(model, np.load(CACHE / "spectra.npy", mmap_mode="r"), device=device), SPECTRUM.path)


if __name__ == "__main__":
    main()
