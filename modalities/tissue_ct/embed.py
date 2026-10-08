"""Embed the cached CPTAC patients (``download.py``): H-Optimus-0 on the slide tiles, CT-FM on the CT volume.

* ``pathology/h_optimus_0``: every tile is normalized with H-Optimus-0's statistics and embedded (timm, bfloat16
  autocast). The tile embeddings are averaged into one vector per slide.
* ``ct/ct_fm``: the volume is z-scored and embedded by the CT-FM SegResNet encoder (bfloat16 autocast). The last
  feature map is averaged over space.

    pixi run -e modalities python -m modalities.tissue_ct.embed
"""

import torch

from modalities.tissue_ct import CACHE, CT, PATHOLOGY, patients

H_OPTIMUS_STATS = ([0.707223, 0.578729, 0.703617], [0.211883, 0.230117, 0.177517])
TILE_BATCH = 64


def h_optimus_0(
    device: str,
) -> torch.nn.Module:
    import timm

    return timm.create_model("hf_hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5).to(device).eval()


def ct_fm(
    device: str,
) -> torch.nn.Module:
    from lighter_zoo import SegResEncoder

    return SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor").to(device).eval()


@torch.inference_mode()
def embed_slide(
    model: torch.nn.Module,
    tiles: torch.Tensor,
    batch_size: int = TILE_BATCH,
) -> torch.Tensor:
    """Mean H-Optimus-0 embedding of the ``[T, 3, 224, 224]`` uint8 tiles of one slide."""
    device = next(model.parameters()).device
    mean, std = (torch.tensor(values).view(1, 3, 1, 1) for values in H_OPTIMUS_STATS)
    outputs = []
    for start in range(0, len(tiles), batch_size):
        images = ((tiles[start : start + batch_size].float() / 255.0) - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs.append(model(images.to(device)).float().cpu())
    return torch.cat(outputs).mean(0)


@torch.inference_mode()
def embed_volume(
    model: torch.nn.Module,
    volume: torch.Tensor,
) -> torch.Tensor:
    """CT-FM embedding of one ``[96, 224, 224]`` volume in [0, 1]."""
    device = next(model.parameters()).device
    volume = volume.float()
    volume = (volume - volume.mean()) / (volume.std() + 1e-6)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        features = model(volume[None, None].to(device))
    features = features[-1] if isinstance(features, (list, tuple)) else features
    return features.mean((2, 3, 4)).float().cpu()[0]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for file, build, embed, key in ((PATHOLOGY, h_optimus_0, embed_slide, "tiles"), (CT, ct_fm, embed_volume, "ct")):
        if file.exists():
            continue
        model = build(device)
        vectors = [embed(model, torch.load(CACHE / f"{patient}.pt", weights_only=False)[key]) for patient in patients()]
        file.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.stack(vectors), file.path)
        print(f"wrote {file.path} {tuple(file.shape)}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
