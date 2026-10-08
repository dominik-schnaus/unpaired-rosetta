"""AstroDINO, the image encoder of AstroCLIP (Parker et al., 2024), rebuilt from ``astrodino.ckpt`` without AstroCLIP.

AstroCLIP builds it through its DINOv2 training setup. The result is a standard DINOv2 ViT-L with the settings of
``astroclip/astrodino/config.yaml`` (12-pixel patches, 144-pixel crops, layer scale 1e-5, 4 block chunks) and the
``teacher`` weights of the checkpoint. The DINOv2 code comes from torch hub. The DINO training head is dropped.
"""

import os
from pathlib import Path

import torch
from torch import nn

ARCHITECTURE = dict(img_size=144, patch_size=12, init_values=1.0e-5, block_chunks=4)


def load_astrodino(
    checkpoint: Path,
) -> nn.Module:
    """AstroDINO backbone in eval mode. ``model(images)`` returns the normalized CLS token."""
    os.environ.setdefault("XFORMERS_DISABLED", "1")  # DINOv2's xFormers attention has no float32 CPU kernel.
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14", pretrained=False, trust_repo=True, **ARCHITECTURE
    )
    teacher = torch.load(checkpoint, map_location="cpu", weights_only=False)["teacher"]
    state = {key.removeprefix("backbone."): value for key, value in teacher.items() if key.startswith("backbone.")}
    model.load_state_dict(state)
    return model.eval()
