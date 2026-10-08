"""Tests that the short code in the notebook and the faster library code give identical results."""

from pathlib import Path
from typing import Any

import nbformat
import pytest
import torch

from unpaired_rosetta.evaluation import disjoint_split, retrieval_metrics
from unpaired_rosetta.randomness import Randomness
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

NOTEBOOK = Path(__file__).parents[1] / "unpaired_rosetta.ipynb"


def notebook_reference() -> dict[str, Any]:
    """Run the notebook cells tagged "reference" (the method and the FOSCTTM) in a fresh namespace."""
    namespace: dict[str, Any] = {}
    exec(  # noqa: S102
        "import torch\nfrom scipy.optimize import linear_sum_assignment\nfrom sklearn.cluster import KMeans\n"
        "from threadpoolctl import threadpool_limits\nfrom unpaired_rosetta.qap import MPOptQAPSolver\n",
        namespace,
    )
    for cell in nbformat.read(NOTEBOOK, as_version=4).cells:
        if "reference" in cell.metadata.get("tags", []):
            exec(cell.source, namespace)  # noqa: S102
    return namespace


def clustered_spaces(
    num_samples: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    centers = 3 * torch.randn(15, 20, generator=generator)
    samples = centers[torch.randint(0, 15, (num_samples,), generator=generator)] + torch.randn(
        num_samples, 20, generator=generator
    )
    rotation = torch.linalg.qr(torch.randn(28, 28, generator=generator))[0][:20]
    return samples, samples @ rotation + 0.3 * torch.randn(num_samples, 28, generator=generator)


def test_the_notebook_has_the_four_hyperparameters_of_the_paper() -> None:
    reference = notebook_reference()
    assert (reference["C"], reference["S"], reference["b"], reference["R"]) == (30, 30, 10_000, 100)


@pytest.mark.parametrize("num_pairs", [0, 7])
def test_notebook_and_library_give_the_same_map(
    num_pairs: int,
) -> None:
    reference = notebook_reference()
    reference.update(C=10, S=4, b=700, R=5)  # small enough for a test, with several read-out blocks
    samples_x, samples_y = clustered_spaces(3000)
    split = Randomness(11)
    indices_x, indices_y = disjoint_split(2800, split)
    pairs = torch.arange(2800, 2800 + num_pairs)
    pairs_x, pairs_y = (samples_x[pairs], samples_y[pairs]) if num_pairs else (None, None)

    notebook_randomness = Randomness(11)
    disjoint_split(2800, notebook_randomness)
    mean_x, mean_y, weight = reference["wasserstein_procrustes"](
        samples_x[indices_x], samples_y[indices_y], pairs_x, pairs_y, notebook_randomness
    )

    library = WassersteinProcrustes(
        num_clusters=10, num_restarts=4, batch_size=700, num_iterations=5, num_workers=2, num_threads=3
    )
    library.fit(samples_x[indices_x], samples_y[indices_y], pairs_x, pairs_y, split)
    assert torch.equal(library.mean_x, mean_x) and torch.equal(library.mean_y, mean_y)
    assert torch.equal(library.weight, weight)


def test_notebook_foscttm_equals_the_library() -> None:
    reference = notebook_reference()
    samples_x, samples_y = clustered_spaces(2500, seed=3)
    aligner = WassersteinProcrustes(num_clusters=10, num_restarts=2, batch_size=1000, num_iterations=2).fit(
        samples_x[:2000], samples_y[:2000]
    )
    val_x, val_y = samples_x[2000:], samples_y[2000:]
    notebook = reference["foscttm"](aligner.similarity, val_x, val_y, Randomness(5), block_size=128)
    library = retrieval_metrics(aligner, val_x, val_y, Randomness(5), block_size=128)["foscttm"]
    assert notebook == library
