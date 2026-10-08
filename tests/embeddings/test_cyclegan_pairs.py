"""Recompute the embeddings of the CycleGAN pairs (Tab. 11) with modalities/cyclegan_pairs and compare them.

The caches must match exactly, because they are deterministic CPU transforms of the raw data. The embeddings of the
first rows of each stored file are compared by cosine similarity, because they were computed on a GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from modalities import cyclegan_pairs
from modalities.cyclegan_pairs import download
from modalities.cyclegan_pairs.embed import Inputs, embed, load_encoder

if TYPE_CHECKING:
    from pathlib import Path

    from modalities.common import EmbeddingFile

NUM_ROWS = 8
MIN_COSINE = 0.999

pytestmark = pytest.mark.embeddings


def require(
    path: Path,
) -> None:
    if not path.exists():
        pytest.skip(f"{path} is missing")


@pytest.mark.parametrize("corpus", ["synthrad2023", "synthrad2023_task2"])
def test_synthrad_slices(
    corpus: str,
) -> None:
    _, task, modalities, _, _ = download.SYNTHRAD[corpus]
    require(download.DATA_ROOT / corpus / task)
    require(download.synthrad_cache(corpus, modalities[0]))
    patients = download.synthrad_patients(corpus)
    assert len(patients) * download.SLICES_PER_PATIENT == 3600
    slices_x, slices_y, _ = download.patient_slices(patients[0], modalities)
    for modality, slices in zip(modalities, (slices_x, slices_y)):
        cached = np.load(download.synthrad_cache(corpus, modality), mmap_mode="r")[: download.SLICES_PER_PATIENT]
        np.testing.assert_array_equal(np.stack(slices), cached)


def test_sen12ms_patches() -> None:
    import pyarrow
    import pyarrow.parquet as parquet

    first_shard = download.DATA_ROOT / "sen12mscr" / next(iter(download.SEN12MS_SHARDS))
    require(first_shard)
    require(download.sen12ms_cache("s1"))
    batch = next(parquet.ParquetFile(first_shard).iter_batches(batch_size=NUM_ROWS, columns=["s1", "s2"]))
    sar, optical, _ = download.decode_patches(pyarrow.Table.from_batches([batch]))
    assert len(sar) == NUM_ROWS
    for sensor, patches in (("s1", sar), ("s2", optical)):
        np.testing.assert_array_equal(
            np.stack(patches), np.load(download.sen12ms_cache(sensor), mmap_mode="r")[:NUM_ROWS]
        )


def test_row_counts() -> None:
    require(download.DATA_ROOT / "LLVIP" / "visible" / "train")
    require(download.arctic_wav_dir("rms"))
    assert len(download.llvip_frame_ids()) == cyclegan_pairs.VISIBLE.shape[0]
    assert len(download.arctic_prompt_ids()) == cyclegan_pairs.SPEAKER_CLB.shape[0]


@pytest.mark.slow
@pytest.mark.parametrize("file", cyclegan_pairs.FILES, ids=lambda file: file.relative_path)
def test_first_rows(
    file: EmbeddingFile,
) -> None:
    require(file.path)
    encoder, transform = load_encoder(file.model)
    try:
        inputs = torch.utils.data.Subset(Inputs(file, transform), range(NUM_ROWS))
    except FileNotFoundError as missing:
        pytest.skip(f"raw data missing: {missing}")
    computed = embed(encoder, inputs, batch_size=NUM_ROWS)
    stored = torch.load(file.path, mmap=True)[:NUM_ROWS].float()
    cosine = torch.nn.functional.cosine_similarity(computed, stored, dim=1)
    print(
        f"{file.relative_path}: min cosine {cosine.min().item():.6f}, "
        f"max abs difference {(computed - stored).abs().max().item():.2e}"
    )
    assert cosine.min().item() > MIN_COSINE
