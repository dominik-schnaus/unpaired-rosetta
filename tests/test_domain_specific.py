"""Tests of the data code and the metrics of the domain-specific benchmarks (Sec. 4.2) on toy inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import scipy.sparse as sp
import torch

from experiments.domain_specific import FixedMap, best_at_precision
from modalities import brain_scans
from modalities.single_cell.cross_species import embed
from modalities.single_cell.prepare import label_indices
from unpaired_rosetta.evaluation import retrieval_metrics
from unpaired_rosetta.randomness import Randomness

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_cell_types_index_the_sorted_names() -> None:
    assert label_indices(["T", "B", "NK", "B"]).tolist() == [2, 0, 1, 0]


def test_materialize_puts_the_same_shared_image_in_the_same_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNPAIRED_ROSETTA_ROOT", str(tmp_path))
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(10, 128, generator=generator)  # one response per shared image, the same for all subjects
    payloads = {}
    for index, subject in enumerate(brain_scans.SUBJECTS):
        seen = torch.arange(10) if index % 2 else torch.arange(1, 10)  # half of the subjects missed image 0
        order = seen[torch.randperm(len(seen), generator=generator)]  # each subject saw them in its own order
        payloads[subject] = {"Z_train": torch.randn(5, 128), "Z_test": images[order], "labels_test": order + 100}
    monkeypatch.setattr(brain_scans, "stage_three_payload", payloads.__getitem__)
    brain_scans.materialize()
    for subject in brain_scans.SUBJECTS:
        validation = torch.load(tmp_path / "embeddings" / brain_scans.VALIDATION_DATASET / subject / "mlp128.pt")
        assert torch.equal(validation, images[1:])  # images seen by every subject, sorted by their NSD id
        training = torch.load(tmp_path / "embeddings" / brain_scans.TRAINING_DATASET / subject / "mlp128.pt")
        assert torch.equal(training, payloads[subject]["Z_train"])


def test_a_fixed_map_that_undoes_a_rotation_is_perfect() -> None:
    generator = torch.Generator().manual_seed(0)
    samples_x = torch.randn(200, 16, generator=generator) + 3.0
    rotation = torch.linalg.qr(torch.randn(16, 16, generator=generator))[0]
    samples_y = samples_x @ rotation
    aligner = FixedMap(rotation, samples_x.mean(dim=0), samples_y.mean(dim=0))
    metrics = retrieval_metrics(aligner, samples_x, samples_y, Randomness(0))
    assert metrics["foscttm"] == 0.0 and metrics["mean_rank"] == 1.0


def test_species_embedding_ignores_the_library_size() -> None:
    generator = torch.Generator().manual_seed(0)
    counts = torch.poisson(3 * torch.rand(300, 2500, generator=generator), generator=generator).numpy()
    deeper = counts.copy()
    deeper[0] *= 2  # the same cell sequenced twice as deep
    embedding, embedding_deeper = embed(sp.csr_matrix(counts)), embed(sp.csr_matrix(deeper))
    assert embedding.shape == (300, 50)
    assert torch.allclose(embedding.abs(), embedding_deeper.abs(), atol=1e-3)


def test_best_values_are_compared_at_the_printed_precision() -> None:
    means = pd.Series({"a": 0.00004, "b": 0.00001, "c": 0.0003})
    assert best_at_precision(means, 4).tolist() == [True, True, False]
    assert best_at_precision(pd.Series([0.91, 0.94]), 3, lower_is_better=False).tolist() == [False, True]
