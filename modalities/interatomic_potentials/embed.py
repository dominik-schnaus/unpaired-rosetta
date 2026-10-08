"""PCA of the cached MACE descriptors: ``embeddings/MP20-structure/<potential>/<potential>-pca256-structure.pt``.

pixi run python -m modalities.interatomic_potentials.embed
"""

import numpy as np
import torch

from modalities.common import EmbeddingFile
from modalities.interatomic_potentials import FILES, descriptor_path
from modalities.pca import fit_pca, project_rows

# GPU batch sizes of the stored files. They reproduce the files bit for bit (see modalities/pca.py).
BATCH_SIZES = {"mace_mp_small": 298, "mace_mp_medium": 1024}


def embed(
    file: EmbeddingFile,
    device: str = "cuda",
) -> torch.Tensor:
    descriptors = np.load(descriptor_path(file.modality), mmap_mode="r")
    basis = fit_pca(descriptors, file.shape[1])
    return project_rows(descriptors, basis, BATCH_SIZES[file.modality], device)


if __name__ == "__main__":
    for file in FILES:
        if file.exists():
            continue
        file.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embed(file), file.path)
        print(f"wrote {file.path}")
