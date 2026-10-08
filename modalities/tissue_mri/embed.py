"""Embed the TCGA glioma patients of ``cohort.csv``: Prov-GigaPath on the slide, DINO ViT-B/16 on the MRI.

* ``pathology/gigapath``: the slide is downloaded from the GDC. 256 tissue patches of 256 x 256 pixels are read
  around random tissue pixels of a 1024-pixel thumbnail, at the level closest to 0.5 um per pixel. They are resized
  to 224 pixels and embedded by the Prov-GigaPath tile encoder (float32). The patch embeddings are averaged and the
  slide is deleted.
* ``mri/dinov1_vitb16-comp``: five axial slices spread over the tumour (BraTS segmentation) are cropped around the
  tumour with a 25 % margin. The t1ce, t2 and flair crops, each min-max scaled, form the three channels of an RGB
  image of 224 x 224 pixels. DINO ViT-B/16 (float32) embeds the five images, which are averaged.

    pixi run -e modalities python -m modalities.tissue_mri.embed
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import torch
from PIL import Image

from modalities.tissue_mri import BRATS_FOLDER, BRATS_ZIP, DATA_ROOT, MRI, PATHOLOGY, cohort

if TYPE_CHECKING:
    from modalities.common import EmbeddingFile

IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
NUM_SLICES, MARGIN, IMAGE_SIZE = 5, 0.25, 224
NUM_PATCHES, PATCH_SIZE, TARGET_MPP = 256, 256, 0.5
COMPOSITE = ("t1ce", "t2", "flair")  # the channels of the RGB composite


def nifti(
    archive: zipfile.ZipFile,
    subject: str,
    sequence: str,
) -> np.ndarray:
    return nib.Nifti1Image.from_bytes(archive.read(f"{BRATS_FOLDER}/{subject}/{subject}_{sequence}.nii")).get_fdata()


def min_max(
    array: np.ndarray,
) -> np.ndarray:
    array = array.astype(np.float32)
    return (array - array.min()) / (np.ptp(array) + 1e-6)


def mri_composites(
    archive: zipfile.ZipFile,
    subject: str,
) -> torch.Tensor:
    """``[5, 3, 224, 224]`` uint8: t1ce/t2/flair crops around the tumour on five axial slices spread over it."""
    segmentation = nifti(archive, subject, "seg")
    volumes = {sequence: nifti(archive, subject, sequence) for sequence in COMPOSITE}
    tumour_slices = np.where((segmentation > 0).sum(axis=(0, 1)) > 0)[0]
    if len(tumour_slices) == 0:
        tumour_slices = np.array([segmentation.shape[2] // 2])
    chosen = tumour_slices[np.linspace(0, len(tumour_slices) - 1, min(NUM_SLICES, len(tumour_slices))).astype(int)]
    composites = []
    for z in chosen:
        rows, columns = tumour_crop(segmentation[:, :, z] > 0)
        channels = []
        for sequence in COMPOSITE:
            crop = Image.fromarray((min_max(volumes[sequence][:, :, z][rows, columns]) * 255).astype(np.uint8))
            channels.append(np.asarray(crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)))
        composites.append(np.stack(channels))
    return torch.from_numpy(np.stack(composites))


def tumour_crop(
    mask: np.ndarray,
) -> tuple[slice, slice]:
    """The tumour's bounding box with a 25 % margin plus 4 pixels, or the whole slice if the tumour is tiny."""
    if mask.sum() < 20:
        return slice(None), slice(None)
    yy, xx = np.where(mask)
    dy, dx = int((yy.max() - yy.min()) * MARGIN) + 4, int((xx.max() - xx.min()) * MARGIN) + 4
    return slice(max(0, yy.min() - dy), yy.max() + dy), slice(max(0, xx.min() - dx), xx.max() + dx)


def dino_vitb16(
    device: str,
) -> torch.nn.Module:
    return (
        torch.hub.load("facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a", "dino_vitb16", verbose=False)
        .to(device)
        .eval()
    )


@torch.inference_mode()
def embed_mri(
    model: torch.nn.Module,
    composites: torch.Tensor,
) -> torch.Tensor:
    """Mean DINO embedding of the composites of one patient."""
    device = next(model.parameters()).device
    mean, std = (torch.tensor(values).view(1, 3, 1, 1) for values in IMAGENET_STATS)
    images = (composites.float() / 255.0 - mean) / std
    return model(images.to(device)).float().cpu().mean(0)


def tissue_patches(
    slide_path: Path,
    seed: int = 0,
) -> list[Image.Image]:
    """Up to 256 tissue patches around random tissue pixels (saturated, neither white nor black) of the thumbnail."""
    import openslide

    slide = openslide.OpenSlide(str(slide_path))
    mpp = float(slide.properties.get("openslide.mpp-x", 0.5) or 0.5)
    hsv = np.asarray(slide.get_thumbnail((1024, 1024)).convert("RGB").convert("HSV"))
    ys, xs = np.where((hsv[:, :, 1] > 20) & (hsv[:, :, 2] < 235) & (hsv[:, :, 2] > 20))
    if len(ys) < 20:
        slide.close()
        return []
    level = min(range(slide.level_count), key=lambda index: abs(mpp * slide.level_downsamples[index] - TARGET_MPP))
    half_patch = int(PATCH_SIZE * slide.level_downsamples[level] // 2)
    scale_y, scale_x = slide.dimensions[1] / hsv.shape[0], slide.dimensions[0] / hsv.shape[1]
    patches = []
    for tries, pixel in enumerate(np.random.default_rng(seed).permutation(len(ys))):
        if len(patches) >= NUM_PATCHES or tries > NUM_PATCHES * 8:
            break
        corner = (max(0, int(xs[pixel] * scale_x) - half_patch), max(0, int(ys[pixel] * scale_y) - half_patch))
        patch = slide.read_region(corner, level, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
        pixels = np.asarray(patch)
        if pixels.mean() < 220 and pixels.std() > 15:
            patches.append(patch)
    slide.close()
    return patches


def gigapath(
    device: str,
) -> torch.nn.Module:
    import timm

    return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device).eval()


@torch.inference_mode()
def embed_patches(
    model: torch.nn.Module,
    patches: list[Image.Image],
    batch_size: int = 128,
) -> torch.Tensor:
    """Mean Prov-GigaPath embedding of the patches of one slide."""
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(*IMAGENET_STATS),
        ]
    )
    device = next(model.parameters()).device
    images = torch.stack([transform(patch) for patch in patches]).to(device)
    return (
        torch.cat([model(images[start : start + batch_size]) for start in range(0, len(images), batch_size)])
        .mean(0)
        .cpu()
    )


def download_slide(
    gdc_file: str,
    path: Path,
    attempts: int = 3,
) -> None:
    """Download a slide from the GDC, starting again when the connection breaks off (large slides often do)."""
    for attempt in range(attempts):
        try:
            urllib.request.urlretrieve(f"https://api.gdc.cancer.gov/data/{gdc_file}", path)
            return
        except urllib.error.ContentTooShortError:
            if attempt == attempts - 1:
                raise


def embed_slide(
    model: torch.nn.Module,
    gdc_file: str,
    folder: Path = DATA_ROOT,
) -> torch.Tensor:
    """Download a slide from the GDC into ``folder``, embed it and delete it."""
    slide_path = Path(folder) / f"_tmp_{gdc_file[:8]}.svs"
    try:
        download_slide(gdc_file, slide_path)
        patches = tissue_patches(slide_path)
        if len(patches) < 8:
            raise RuntimeError(f"only {len(patches)} tissue patches in slide {gdc_file}")
        return embed_patches(model, patches)
    finally:
        slide_path.unlink(missing_ok=True)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patients = cohort()
    if not MRI.exists():
        model = dino_vitb16(device)
        with zipfile.ZipFile(BRATS_ZIP) as archive:
            vectors = [embed_mri(model, mri_composites(archive, subject)) for subject in patients["brats"]]
        save(MRI, torch.stack(vectors))
    if not PATHOLOGY.exists():
        model = gigapath(device)
        save(PATHOLOGY, torch.stack([embed_slide(model, gdc_file) for gdc_file in patients["gdc_file"]]))


def save(
    file: EmbeddingFile,
    vectors: torch.Tensor,
) -> None:
    file.path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, file.path)
    print(f"wrote {file.path} {tuple(vectors.shape)}")


if __name__ == "__main__":
    main()
