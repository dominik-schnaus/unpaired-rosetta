"""Problems of the low-rank Gromov-Wasserstein ablation (Table 2), after the experiments of Scetbon et al. (2022).

Every problem is a pair ``(X, Y)`` of 1000 samples with a known correspondence: ``Y[permutation[i]]`` is the partner
of ``X[i]``.

* ``blobs_k10``, ``blobs_k30``, ``curve``, ``uniform``: synthetic point clouds. ``Y`` is ``X`` embedded into 15
  dimensions by a random linear isometry and shuffled. We wrote these ourselves after the toy examples of Scetbon
  et al. (anisotropic Gaussian blobs, a curve, uniform samples). We do not use their generators, because the
  official code (``toy_examples.py`` of github.com/meyerscetbon/LinearGromov) draws two unrelated clouds without a
  ground-truth match (a 2-D/3-D spiral, mixtures of two and three Gaussians).
* ``snare`` (SNARE-seq, RNA and ATAC of the same 1047 cells) and ``splatter`` (a simulated single-cell pair) are
  the single-cell problems distributed with SCOT (Demetci et al., 2020), rows scaled to unit length as in SCOT and
  subsampled to 1000 cells.
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from unpaired_rosetta.settings import raw_root

NAMES = {
    "blobs_k10": "blobs $k$=10",
    "blobs_k30": "blobs $k$=30",
    "curve": "curve",
    "snare": "SNARE-seq",
    "splatter": "splatter",
    "uniform": "uniform",
}

# The SCOT repository at the commit whose data files we use, with their SHA-256 checksums.
SCOT_URL = "https://raw.githubusercontent.com/rsinghlab/SCOT/14649be6e14017dcfe7ba619091b33d1df55f6a9/data"
SCOT_FILES = {
    "SNARE/SNAREseq_rna_feat.npy": "a8a2c2bbfc036dea5b350d01dd2180b93bb3b1c0008bed49493080348da96a9b",
    "SNARE/SNAREseq_atac_feat.npy": "80d685b6517c100a7fd612846ad38209b64ed9caf095d1630af5a1d7797bd79c",
    "simulations/splatter_X.npy": "d1b3497d7e870ebdffa877d5ec3a9b22573ce8558e5dad82d25d883501e5c7be",
    "simulations/splatter_y.npy": "6353802f7664457668e0b84ba322c39b4b1216382f9bcff3c184b039b4a1cd94",
}


def scot_file(
    name: str,
) -> Path:
    """Path of a SCOT data file, downloaded (and checked against its checksum) on first use."""
    path = raw_root() / "SCOT" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        urllib.request.urlretrieve(f"{SCOT_URL}/{name}", temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != SCOT_FILES[name]:
            temporary.unlink()
            raise RuntimeError(f"Checksum mismatch for {name}: {digest}")
        temporary.replace(path)
    return path


def isometric_embedding(
    samples: Tensor,
    dim_out: int,
    generator: torch.Generator,
) -> Tensor:
    """Embed into ``dim_out`` dimensions with a random linear isometry (all pairwise distances are preserved)."""
    matrix = torch.randn(dim_out, samples.shape[1], generator=generator, dtype=samples.dtype)
    left, _, right_transposed = torch.linalg.svd(matrix, full_matrices=False)
    return samples @ (left @ right_transposed).T


def shuffle(
    samples: Tensor,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """The rows in a random order, and where every original row went."""
    order = torch.randperm(samples.shape[0], generator=generator)
    return samples[order], order.argsort()


def blobs(
    num_samples: int,
    num_clusters: int,
    seed: int,
    dim_x: int = 10,
    dim_y: int = 15,
    separation: float = 6.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Anisotropic Gaussian blobs of (almost) equal size."""
    generator = torch.Generator().manual_seed(seed)
    centers = separation * torch.randn(num_clusters, dim_x, generator=generator, dtype=torch.float64)
    sizes = torch.full((num_clusters,), num_samples // num_clusters)
    sizes[: num_samples - sizes.sum()] += 1
    parts = []
    for center, size in zip(centers, sizes.tolist()):
        noise = torch.randn(size, dim_x, generator=generator, dtype=torch.float64)
        anisotropy = torch.randn(dim_x, dim_x, generator=generator, dtype=torch.float64)
        parts.append(noise @ anisotropy / dim_x**0.5 + center)
    samples_x = torch.cat(parts)
    samples_y, permutation = shuffle(isometric_embedding(samples_x, dim_y, generator), generator)
    return samples_x, samples_y, permutation


def uniform(
    num_samples: int,
    seed: int,
    dim_x: int = 10,
    dim_y: int = 15,
) -> tuple[Tensor, Tensor, Tensor]:
    """Uniform samples in the unit cube: no cluster structure."""
    generator = torch.Generator().manual_seed(seed)
    samples_x = torch.rand(num_samples, dim_x, generator=generator, dtype=torch.float64)
    samples_y, permutation = shuffle(isometric_embedding(samples_x, dim_y, generator), generator)
    return samples_x, samples_y, permutation


def curve(
    num_samples: int,
    seed: int,
    dim_y: int = 15,
) -> tuple[Tensor, Tensor, Tensor]:
    """A noisy planar spiral."""
    generator = torch.Generator().manual_seed(seed)
    angle = torch.linspace(0, 4 * np.pi, num_samples, dtype=torch.float64)
    samples_x = torch.stack([angle * torch.cos(angle), angle * torch.sin(angle)], dim=1) / (4 * np.pi)
    samples_x = samples_x + 0.01 * torch.randn(samples_x.shape, generator=generator, dtype=torch.float64)
    samples_y, permutation = shuffle(isometric_embedding(samples_x, dim_y, generator), generator)
    return samples_x, samples_y, permutation


def single_cell(
    name: str,
    num_samples: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """SNARE-seq or Splatter with unit-length rows, subsampled to ``num_samples`` cells."""
    files = (
        ("SNARE/SNAREseq_rna_feat.npy", "SNARE/SNAREseq_atac_feat.npy")
        if name == "snare"
        else ("simulations/splatter_X.npy", "simulations/splatter_y.npy")
    )
    samples_x, samples_y = (np.load(scot_file(file)) for file in files)
    if samples_x.shape[0] > num_samples:
        indices = np.random.RandomState(seed).choice(samples_x.shape[0], num_samples, replace=False)
        samples_x, samples_y = samples_x[indices], samples_y[indices]
    samples_x = torch.as_tensor(
        samples_x / np.linalg.norm(samples_x, axis=1, keepdims=True).clip(min=1e-12), dtype=torch.float64
    )
    samples_y = torch.as_tensor(
        samples_y / np.linalg.norm(samples_y, axis=1, keepdims=True).clip(min=1e-12), dtype=torch.float64
    )
    samples_y, permutation = shuffle(samples_y, torch.Generator().manual_seed(seed))
    return samples_x, samples_y, permutation


def load(
    name: str,
    seed: int,
    num_samples: int = 1000,
) -> tuple[Tensor, Tensor, Tensor]:
    """``(X, Y, permutation)`` of a problem. The seed draws the data, and for the cells also the subsample and order."""
    if name == "blobs_k10":
        return blobs(num_samples, 10, seed)
    if name == "blobs_k30":
        return blobs(num_samples, 30, seed)
    if name == "curve":
        return curve(num_samples, seed)
    if name == "uniform":
        return uniform(num_samples, seed)
    if name in ("snare", "splatter"):
        return single_cell(name, num_samples, seed)
    raise ValueError(f"Unknown problem {name!r}; expected one of {list(NAMES)}.")
