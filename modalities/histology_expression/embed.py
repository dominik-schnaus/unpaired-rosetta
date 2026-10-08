"""Embed the cached spots (``prepare.py``): H-Optimus-0 on the tiles, PCA 64 on the expression.

* ``tile/h_optimus_0``: every 224 x 224 tile scaled to [0, 1], normalized with H-Optimus-0's own statistics and
  embedded in float32 (the class-token output of timm's model).
* ``expression/expression-pca64-ccrcc``: the leading 64 principal components of the expression matrix
  (``modalities/pca.py``).

    pixi run -e modalities python -m modalities.histology_expression.embed
"""

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import trange

from modalities.histology_expression import CACHE, EXPRESSION, TILE
from modalities.pca import fit_pca, project_rows

H_OPTIMUS_STATS = ([0.707223, 0.578729, 0.703617], [0.211883, 0.230117, 0.177517])
TILE_BATCH = 64
# GPU batch size of the stored expression file. It reproduces the file bit for bit (see modalities/pca.py).
EXPRESSION_BATCH = 1024


def h_optimus_0(
    device: str,
) -> torch.nn.Module:
    import timm

    return timm.create_model("hf_hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5).to(device).eval()


def tile_batch(
    tiles: np.ndarray,
) -> Tensor:
    """``[B, 224, 224, 3]`` uint8 tiles to normalized ``[B, 3, 224, 224]`` float32 images."""
    mean, std = (torch.tensor(values).view(1, 3, 1, 1) for values in H_OPTIMUS_STATS)
    images = torch.from_numpy(np.array(tiles, dtype=np.float32)).permute(0, 3, 1, 2) / 255.0
    return (images - mean) / std


@torch.inference_mode()
def embed_tiles(
    model: torch.nn.Module,
    tiles: np.ndarray,
    batch_size: int = TILE_BATCH,
) -> Tensor:
    """H-Optimus-0 embeddings ``[N, 1536]`` of the uint8 tiles ``[N, 224, 224, 3]``."""
    device = next(model.parameters()).device
    batches = [
        model(tile_batch(tiles[start : start + batch_size]).to(device)).float().cpu()
        for start in trange(0, len(tiles), batch_size, leave=False)
    ]
    return torch.cat(batches)


def embed_expression(
    device: str,
) -> Tensor:
    expression = np.load(CACHE / "expression.npy", mmap_mode="r")
    return project_rows(expression, fit_pca(expression, EXPRESSION.shape[1]), EXPRESSION_BATCH, device)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for file in (TILE, EXPRESSION):
        if file.exists():
            continue
        if file is TILE:
            embeddings = embed_tiles(h_optimus_0(device), np.load(CACHE / "tiles.npy", mmap_mode="r"))
        else:
            embeddings = embed_expression(device)
        file.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, file.path)
        print(f"wrote {file.path} {tuple(embeddings.shape)}")


if __name__ == "__main__":
    main()
