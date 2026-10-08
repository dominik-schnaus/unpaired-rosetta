"""Loading of stored embeddings.

A file holds a bfloat16 tensor ``[N, d]`` or ``[N, T, d]`` (several texts per item, padded with NaN) at
``<root>/embeddings/<dataset>/<modality>/<model>.pt``.
"""

from pathlib import Path

import torch
from torch import Tensor
from torch.nn.functional import normalize

from unpaired_rosetta.settings import storage_root  # noqa: F401 (imported from here by many modules)


def embedding_path(
    dataset: str,
    modality: str,
    model: str,
    root: Path | None = None,
) -> Path:
    return (storage_root() if root is None else Path(root)) / "embeddings" / dataset / modality / f"{model}.pt"


def reduce_texts(
    embeddings: Tensor,
) -> Tensor:
    """One unit-length embedding per item: normalize each text, average over texts (skipping NaN), normalize again.

    Runs in the stored precision, as for the numbers in the paper.
    """
    if embeddings.ndim == 3 and embeddings.shape[1] == 1:
        embeddings = embeddings[:, 0]
    elif embeddings.ndim == 3:
        embeddings = normalize(embeddings, dim=-1).nanmean(dim=1)
    return normalize(embeddings, dim=-1)


def num_rows(
    path: Path,
) -> int:
    return torch.load(path, mmap=True).shape[0]


def load_rows(
    path: Path,
    indices: Tensor | None = None,
) -> Tensor:
    """Rows ``indices`` of a stored file as unit-length float32 embeddings.

    The file is memory-mapped and read in file order, so only the selected rows are loaded.
    """
    embeddings = torch.load(path, mmap=True)
    if indices is not None:
        order = torch.argsort(indices)
        embeddings = embeddings[indices[order]][torch.argsort(order)]
    return reduce_texts(embeddings).float()
