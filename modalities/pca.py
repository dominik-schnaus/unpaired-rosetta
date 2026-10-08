"""PCA of a cached feature matrix. It encodes the tabular side of several modality pairs (MACE descriptors of
MP-20, Cell Painting and L1000 profiles, spot expression of HEST).

A PCA basis is a rotation of the centred data followed by a truncation, so it adds no learned structure. An
orthogonal aligner cannot see the rotation. The basis is fitted once on at most 20,000 rows (a fixed random subset
drawn with seed 0). The rows are centred but not standardized, and the components are the leading right singular
vectors. A row is embedded as ``(row - mean) @ components^T``.

Two details make the result reproducible bit for bit. The last bits of the singular vectors depend on the number of
SVD threads (MKL), so the fit always uses the 24 threads the stored bases were fitted with. The last bits of the
float32 products depend on the batch shape. The stored files were projected on the GPU in batches sized by the free
GPU memory at the time, and the modality packages record these sizes. With both, ``project_rows`` reproduces the
stored files exactly (checked on an RTX 4500 Ada).
"""

import numpy as np
import torch
from torch import Tensor

FIT_ROWS = 20_000
FIT_THREADS = 24


def fit_pca(
    features: np.ndarray,
    num_components: int,
) -> dict[str, Tensor]:
    """Mean and the ``num_components`` leading principal axes (``[num_components, d]``) of ``features`` (``[N, d]``)."""
    rows = np.arange(len(features))
    if len(rows) > FIT_ROWS:
        rows = np.sort(np.random.default_rng(0).permutation(len(rows))[:FIT_ROWS])
    sample = torch.from_numpy(np.asarray(features[rows], dtype="float32")).reshape(len(rows), -1)
    mean = sample.mean(0)
    threads = torch.get_num_threads()
    torch.set_num_threads(FIT_THREADS)
    try:
        _, _, right = torch.linalg.svd(sample - mean, full_matrices=False)
    finally:
        torch.set_num_threads(threads)
    if right.shape[0] < num_components:
        raise ValueError(f"{len(rows)} rows give at most {right.shape[0]} components, {num_components} requested.")
    return {"mean": mean, "components": right[:num_components].contiguous()}


def project(
    features: Tensor,
    basis: dict[str, Tensor],
) -> Tensor:
    """Coordinates of ``features`` (``[B, d]``) in the basis of ``fit_pca``."""
    return (features.flatten(1) - basis["mean"]) @ basis["components"].T


def project_rows(
    features: np.ndarray,
    basis: dict[str, Tensor],
    batch_size: int,
    device: str = "cuda",
) -> Tensor:
    """``project`` of every row of ``features``, in consecutive batches of ``batch_size`` rows on ``device``."""
    basis = {name: value.to(device) for name, value in basis.items()}
    batches = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(np.array(features[start : start + batch_size], dtype="float32"))
        batches.append(project(batch.to(device), basis).cpu())
    return torch.cat(batches)
