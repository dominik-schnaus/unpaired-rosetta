"""Compare our mini-vec2vec with the official notebook (github.com/guy-dar/mini-vec2vec, ``linear_vec2vec.ipynb``).

The tests compare the functions of the notebook stage by stage and run its training cells on a small problem with the
same random choices as our port. Its global generators get the same seeds as our ``Randomness``. Its two sources of
fresh randomness are ``KMeans()`` without a seed and the scipy 2-opt, which draws from an unseeded generator. Both are
replaced by the same seeded draws that our port makes. Run ``tests/mini_vec2vec/clone.sh`` first.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import nbformat
import numpy as np
import pytest
import sklearn.cluster
import torch
from scipy.optimize import quadratic_assignment
from threadpoolctl import threadpool_limits

from unpaired_rosetta.baselines.mini_vec2vec import MiniVec2Vec, cosine_neighbors, mean_of_neighbors, two_opt
from unpaired_rosetta.determinism import KMEANS_THREADS
from unpaired_rosetta.geometric_initialization import centered_kernel
from unpaired_rosetta.linalg import center_and_normalize, polar
from unpaired_rosetta.randomness import Randomness

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

OFFICIAL = Path(__file__).parent / "official"
pytestmark = [pytest.mark.official, pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/mini_vec2vec/clone.sh")]

SEED = 5


@pytest.fixture(scope="module")
def notebook() -> nbformat.NotebookNode:
    return nbformat.read(OFFICIAL / "linear_vec2vec.ipynb", as_version=4)


def definitions(
    notebook: nbformat.NotebookNode,
) -> dict:
    """The imports and function definitions of the notebook, without running its download and training cells."""
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            module = ast.parse(cell.source)
            module.body = [
                node for node in module.body if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom))
            ]
            exec(compile(module, "<notebook>", "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def official(
    notebook: nbformat.NotebookNode,
) -> dict[str, object]:
    return definitions(notebook)


def training_cells(
    notebook: nbformat.NotebookNode,
    replacements: dict[str, str],
) -> str:
    """Source of the training cells (anchors, read-out, refine1, refine2) with smaller sizes for a quick test."""
    anchor_cell = next(
        i for i, cell in enumerate(notebook.cells) if "all_centers1, all_centers2 = [], []" in cell.source
    )
    refine2_cell = next(i for i, cell in enumerate(notebook.cells) if "KMeans(n_clusters=500)" in cell.source)
    source = "\n".join(
        cell.source for cell in notebook.cells[anchor_cell : refine2_cell + 1] if cell.cell_type == "code"
    )
    for old, new in replacements.items():
        assert old in source, old
        source = source.replace(old, new)
    return source


def seeded_kmeans(
    n_clusters: int,
    init: str | np.ndarray = "k-means++",
) -> sklearn.cluster.KMeans:
    """``KMeans`` with a seed drawn from the global NumPy generator, as in our port."""
    return sklearn.cluster.KMeans(n_clusters=n_clusters, init=init, random_state=int(np.random.randint(0, 2**31 - 1)))


def seeded_two_opt(
    kernel1: np.ndarray,
    kernel2: np.ndarray,
    method: str,
    options: dict[str, object],
) -> OptimizeResult:
    """The scipy 2-opt from a random permutation whose seed comes from the global torch generator, as in our port."""
    seed = torch.randint(0, 2**32 - 1, (1,)).item()
    start = np.random.RandomState(seed).permutation(kernel1.shape[0])
    guess = np.column_stack([np.arange(len(start)), start])
    return quadratic_assignment(
        kernel1, kernel2, method=method, options={"maximize": options["maximize"], "partial_guess": guess}
    )


@pytest.fixture
def problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two centered, normalized sets related by a known rotation. The notebook's Procrustes needs equal widths."""
    generator = torch.Generator().manual_seed(0)
    centers = 3 * torch.randn(20, 16, generator=generator)
    labels = torch.randint(0, 20, (600,), generator=generator)
    x = centers[labels] + 0.5 * torch.randn(600, 16, generator=generator)
    rotation = polar(torch.randn(16, 16, generator=generator))
    order = torch.randperm(600, generator=generator)
    return x, (x @ rotation)[order], torch.argsort(order)  # x[i] belongs to y[partner[i]]


def test_cluster_kernel_equals_official_sim(
    official: dict[str, object],
    problem: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    centers = problem[0][:20]
    assert torch.allclose(centered_kernel(centers), official["sim"](centers, centers), atol=1e-4)


@pytest.mark.parametrize("seed", range(5))
def test_two_opt_equals_scipy(
    problem: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    seed: int,
) -> None:
    """Our 2-opt equals the scipy 2-opt (maximize) that the notebook uses, from the same start."""
    x, y, _ = problem
    kernel1 = centered_kernel(x[:20]).double().numpy()
    kernel2 = centered_kernel(y[seed : seed + 20]).double().numpy()
    start = np.random.RandomState(seed).permutation(20)
    permutation, score = two_opt(kernel1, kernel2, start)
    guess = np.column_stack([np.arange(20), start])
    result = quadratic_assignment(kernel1, kernel2, method="2opt", options={"maximize": True, "partial_guess": guess})
    np.testing.assert_array_equal(permutation, result.col_ind)
    assert score == result.fun


def test_readout_equals_official(
    official: dict[str, object],
    problem: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x, y = (center_and_normalize(samples, samples.mean(dim=0, keepdim=True)) for samples in problem[:2])
    anchors_x, anchors_y = x[:40] + 0.1, y[:40] - 0.1  # any anchors, not normalized, like k-means centers
    ours = MiniVec2Vec().readout(x, y, anchors_x, anchors_y)

    similarity = official["cos_sim_matrix"](
        official["cos_sim_matrix"](x, anchors_x), official["cos_sim_matrix"](y, anchors_y)
    )
    matched = y[similarity.topk(dim=-1, k=50).indices].transpose(-1, -2) @ (torch.ones(50) / 50)
    assert torch.allclose(ours, official["train_orthogonal_linear"](x, matched), atol=1e-5)


def test_neighbors_and_means_equal_official(
    official: dict[str, object],
    problem: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x, y, _ = problem
    ours = cosine_neighbors(x, y, 50, block_size=128)
    theirs = official["cos_sim_matrix"](x, y).topk(dim=-1, k=50).indices
    assert torch.equal(ours, theirs)
    assert torch.allclose(mean_of_neighbors(ours, y), y[theirs].mean(dim=1), atol=1e-6)


def test_whole_fit_equals_official_notebook(
    notebook: nbformat.NotebookNode,
    problem: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Our fit equals the training cells of the notebook (anchors, read-out, refine1, refine2).

    Both use the same random choices.
    """
    x, y, _ = problem
    ours = MiniVec2Vec(num_runs=3, num_restarts=5, refine1_iterations=10, refine1_samples=200, refine2_clusters=40)
    ours.fit(x, y, randomness=Randomness(SEED))

    source = training_cells(
        notebook,
        {
            "for i in trange(30)": "for i in trange(3)",
            "n_runs=30": "n_runs=5",
            "n_iters = 100": "n_iters = 10",
            "[:1000]": "[:200]",
            "n_clusters=500": "n_clusters=40",
        },
    )
    refine2 = source.index("n_iters = 1\n")
    namespace = definitions(notebook)  # the notebook functions look up these two names here
    namespace.update(KMeans=seeded_kmeans, quadratic_assignment=seeded_two_opt)
    namespace["X_train"] = namespace["X_eval"] = center_and_normalize(x, x.mean(dim=0, keepdim=True))
    namespace["Y_train"] = namespace["Y_eval"] = center_and_normalize(y, y.mean(dim=0, keepdim=True))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    with threadpool_limits(KMEANS_THREADS), torch.inference_mode():
        exec(source[:refine2], namespace)
        # Our refine2 clusters shuffled copies of both sets. The notebook clusters them in their stored order.
        namespace["X_train"] = namespace["X_train"][torch.randperm(600)]
        namespace["Y_train"] = namespace["Y_train"][torch.randperm(600)]
        exec(source[refine2:], namespace)
    assert torch.allclose(ours.weight, namespace["W"], atol=1e-5)


def test_recovers_planted_rotation(
    problem: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x, y, partner = problem
    aligner = MiniVec2Vec(num_runs=3, num_restarts=5, refine1_iterations=10, refine1_samples=200, refine2_clusters=40)
    aligner.fit(x, y, randomness=Randomness(SEED))
    accuracy = (aligner.similarity(x, y).argmax(dim=1) == partner).float().mean()
    assert accuracy > 0.9
