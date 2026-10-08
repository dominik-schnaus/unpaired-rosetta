"""PCA of the consensus profiles: ``embeddings/Rosetta-LINCS-Pilot1/{morphology,expression}/<assay>-pca<k>.pt``.

pixi run python -m modalities.perturbation_profiles.embed
"""

import numpy as np
import torch

from modalities.common import EmbeddingFile
from modalities.pca import fit_pca, project_rows
from modalities.perturbation_profiles import FILES, profile_path

# GPU batch sizes of the stored files. They reproduce the files bit for bit (see modalities/pca.py).
BATCH_SIZES = {"morphology": 541, "expression": 1024}


def embed(
    file: EmbeddingFile,
    device: str = "cuda",
) -> torch.Tensor:
    profiles = np.load(profile_path(file.modality), mmap_mode="r")
    basis = fit_pca(profiles, file.shape[1])
    return project_rows(profiles, basis, BATCH_SIZES[file.modality], device)


if __name__ == "__main__":
    for file in FILES:
        if file.exists():
            continue
        file.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embed(file), file.path)
        print(f"wrote {file.path}")
