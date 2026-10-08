"""Embed the five pairs: every image-like side by a self-supervised vision transformer, the speakers by WavLM-Large.

    pixi run -e modalities python -m modalities.cyclegan_pairs.embed

Each pair uses one encoder on both sides. The vision encoders average all tokens of their last layer. For DINO
ViT-B/16 this is the normed output of its last block, for DINOv2 the class and patch tokens after the final norm.
WavLM-Large averages its last hidden states over time, from 4 s of audio per prompt. Short waveforms are tiled
instead of zero padded, because padding would make the mean depend on the utterance length, which both speakers
share. The audio is then center-cropped and standardized. Everything runs in float32, as for the stored files.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from tqdm.auto import tqdm

from modalities import cyclegan_pairs
from modalities.common import EmbeddingFile
from modalities.cyclegan_pairs.download import (
    DATA_ROOT,
    arctic_prompt_ids,
    arctic_wav_dir,
    llvip_frame_ids,
    sen12ms_cache,
    synthrad_cache,
)
from modalities.image_captions.encoders import DINOV2_HUB, TokenMean, bicubic_transform, dinov2_transform, hub_model

if TYPE_CHECKING:
    from pathlib import Path

DINO_HUB = "facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a"
WAVLM = "microsoft/wavlm-large"
SAMPLE_RATE = 16_000
AUDIO_SECONDS = 4.0
BATCH_SIZE = 32


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LastLayerMean(nn.Module):
    """DINO (v1): the mean over the class and patch tokens of its last block after the final norm."""

    def __init__(
        self,
        backbone: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        return self.backbone.get_intermediate_layers(images, n=1)[0].mean(dim=1)


class SpeechMean(nn.Module):
    """A self-supervised speech encoder with its last hidden states averaged over time.

    The input is ``[B, T]`` at 16 kHz.
    """

    def __init__(
        self,
        name: str,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(name)
        self.backbone.config.apply_spec_augment = False  # masking only runs in training mode anyway

    def forward(
        self,
        waveforms: Tensor,
    ) -> Tensor:
        return self.backbone(waveforms).last_hidden_state.mean(dim=1)


def fixed_length_waveform(
    waveform: Tensor,
    num_samples: int = int(AUDIO_SECONDS * SAMPLE_RATE),
) -> Tensor:
    """Tile a short waveform, take the centered ``num_samples``, and standardize it."""
    waveform = waveform.flatten().float()
    if waveform.numel() < num_samples:
        waveform = waveform.repeat(num_samples // waveform.numel() + 1)
    offset = (waveform.numel() - num_samples) // 2
    waveform = waveform[offset : offset + num_samples]
    return (waveform - waveform.mean()) / (waveform.std() + 1e-7)


def load_encoder(
    model: str,
) -> tuple[nn.Module, Callable]:
    """The encoder of a stored model name and the transform of its inputs (a PIL image or a waveform)."""
    if model == cyclegan_pairs.DINOV1_B16:
        return LastLayerMean(hub_model(DINO_HUB, "dino_vitb16")), bicubic_transform()
    if model == cyclegan_pairs.DINOV2_B14:
        return TokenMean(hub_model(DINOV2_HUB, "dinov2_vitb14"), "x_norm_regtokens"), dinov2_transform()
    if model == cyclegan_pairs.DINOV2_L14:
        return TokenMean(hub_model(DINOV2_HUB, "dinov2_vitl14"), "x_norm_regtokens"), dinov2_transform()
    if model == cyclegan_pairs.WAVLM_LARGE:
        return SpeechMean(WAVLM), fixed_length_waveform
    raise ValueError(f"Unknown model {model!r}")


def grey_image(
    array: np.ndarray,
) -> Image.Image:
    """A uint8 slice as an RGB image (the grey value in all three channels)."""
    return Image.fromarray(np.asarray(array)).convert("RGB")


def rgb_image(
    path: Path,
) -> Image.Image:
    return Image.open(path).convert("RGB")


def patch_image(
    patch: np.ndarray,
) -> Image.Image:
    return Image.fromarray(np.asarray(patch))


def read_wav(
    path: Path,
) -> Tensor:
    from scipy.io import wavfile

    _, samples = wavfile.read(path)
    return torch.from_numpy(samples.astype(np.float32) / 32768.0)


class Inputs(torch.utils.data.Dataset):
    """The raw inputs of one stored file. Row i of both sides of a pair is the same slice, frame, patch or prompt.

    Slices and patches come from the caches, frames and recordings from the corpus.
    """

    def __init__(
        self,
        file: EmbeddingFile,
        transform: Callable,
    ) -> None:
        self.file, self.transform = file, transform
        if file.dataset == cyclegan_pairs.MR_CT_DATASET:
            self.items, self.read = np.load(synthrad_cache("synthrad2023", file.modality), mmap_mode="r"), grey_image
        elif file.dataset == cyclegan_pairs.CBCT_CT_DATASET:
            self.items, self.read = (
                np.load(synthrad_cache("synthrad2023_task2", file.modality), mmap_mode="r"),
                grey_image,
            )
        elif file.dataset == cyclegan_pairs.RGB_THERMAL_DATASET:
            folder = DATA_ROOT / "LLVIP" / file.modality / "train"
            self.items = [folder / f"{frame}.jpg" for frame in llvip_frame_ids()]
            self.read = rgb_image
        elif file.dataset == cyclegan_pairs.SAR_OPTICAL_DATASET:
            self.items, self.read = np.load(sen12ms_cache(file.modality), mmap_mode="r"), patch_image
        elif file.dataset == cyclegan_pairs.SPEAKER_DATASET:
            self.items = [arctic_wav_dir(file.modality) / f"{prompt}.wav" for prompt in arctic_prompt_ids()]
            self.read = read_wav
        else:
            raise ValueError(f"Unknown dataset {file.dataset!r}")

    def __len__(
        self,
    ) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int,
    ) -> Tensor:
        return self.transform(self.read(self.items[index]))


@torch.inference_mode()
def embed(
    encoder: nn.Module,
    inputs: torch.utils.data.Dataset,
    batch_size: int = BATCH_SIZE,
) -> Tensor:
    """Float32 embeddings ``[len(inputs), d]``."""
    encoder = encoder.to(device()).eval()
    loader = torch.utils.data.DataLoader(inputs, batch_size=batch_size, num_workers=2)
    return torch.cat([encoder(batch.to(device())).float().cpu() for batch in tqdm(loader, leave=False)])


def compute_file(
    file: EmbeddingFile,
) -> None:
    encoder, transform = load_encoder(file.model)
    inputs = Inputs(file, transform)
    assert len(inputs) == file.shape[0], (
        f"{file.relative_path}: {len(inputs)} items, the registry expects {file.shape[0]}"
    )
    embeddings = embed(encoder, inputs)
    file.path.parent.mkdir(parents=True, exist_ok=True)
    partial = file.path.with_name(file.path.name + ".partial")
    torch.save(embeddings, partial)
    partial.rename(file.path)
    print(f"wrote {file.path} {tuple(embeddings.shape)}")


if __name__ == "__main__":
    for file in cyclegan_pairs.FILES:
        if not file.exists():
            compute_file(file)
