"""Shared helpers of the modality packages: the description of a stored embedding file and how to fetch it.

Each modality package lists its embedding files as ``EmbeddingFile`` entries. From them the notebook and the scripts
know which files an experiment needs, which are missing and how large they are, without reading any data. A missing
file is either downloaded from the Hugging Face Hub or computed from the raw data by its modality package.
"""

import hashlib
import urllib.request
from dataclasses import dataclass
from math import prod
from pathlib import Path

from unpaired_rosetta.embeddings import embedding_path, storage_root

# Precomputed embeddings of the paper (the exact files behind its numbers).
HF_REPOSITORY = "schnaus/unpaired-rosetta-embeddings"

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}


@dataclass(frozen=True)
class EmbeddingFile:
    dataset: str
    modality: str
    model: str
    shape: tuple[int, ...]  # [items, dim] or [items, texts per item, dim]
    dtype: str = "bfloat16"

    @property
    def path(
        self,
    ) -> Path:
        return embedding_path(self.dataset, self.modality, self.model)

    @property
    def relative_path(
        self,
    ) -> str:
        return f"embeddings/{self.dataset}/{self.modality}/{self.model}.pt"

    @property
    def num_bytes(
        self,
    ) -> int:
        return prod(self.shape) * DTYPE_BYTES[self.dtype]

    def exists(
        self,
    ) -> bool:
        return self.path.exists()


@dataclass(frozen=True)
class RawData:
    """A raw dataset that must be downloaded before its embeddings can be computed."""

    name: str
    download_bytes: int  # what is transferred
    stored_bytes: int  # what stays on disk after unpacking


def format_bytes(
    num_bytes: float,
) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1000 or unit == "TB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1000


def download_precomputed(
    relative_paths: list[str],
) -> None:
    """Download precomputed files (embeddings, labels) from the Hugging Face Hub into the storage root. The paths
    are relative to the storage root."""
    from huggingface_hub import hf_hub_download

    for relative_path in relative_paths:
        hf_hub_download(HF_REPOSITORY, relative_path, repo_type="dataset", local_dir=storage_root())


def download(
    url: str,
    path: Path,
    sha256: str,
) -> Path:
    """Download ``url`` to ``path`` unless it is there already, and check the file against its SHA-256 checksum."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.download")
    urllib.request.urlretrieve(url, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest != sha256:
        temporary.unlink()
        raise RuntimeError(f"Checksum mismatch for {url}: {digest}")
    temporary.replace(path)
    return path
