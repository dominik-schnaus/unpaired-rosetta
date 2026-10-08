"""Compare the mini-vec2vec components of the ablation (``unpaired_rosetta/ablation``) with the official notebook
(github.com/guy-dar/mini-vec2vec, ``linear_vec2vec.ipynb``). Run ``tests/mini_vec2vec_components/clone.sh`` first.

The notebook cells run as they are, except that its unseeded random choices (``KMeans()`` and the start of the scipy
2-opt) are replaced by the seeded draws of our code. Where the runs behind Tables 3 and 4 differ from the notebook,
the test names the difference and checks everything else:

* 2-opt cluster matching (Table 1): identical anchors.
* Relative-representation read-out (Table 3): identical for unit-length anchors. The runs behind Table 3 used the
  matched centers without normalizing them, and the notebook normalizes them (``cos_sim_matrix``).
* refine 1 and refine 2 (Table 4): identical Procrustes steps. The runs behind Table 4 draw 1024 instead of 1000
  samples per refine-1 step and project each moving average back onto the orthogonal matrices.
* k-means: the notebook passes torch tensors to scikit-learn, which clusters them in float64. We cluster in float32,
  like the runs of the paper. The centers differ in the last bits, and so can the clusterings.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Self

import nbformat
import numpy as np
import pytest
import sklearn.cluster
import torch
from scipy.optimize import quadratic_assignment
from threadpoolctl import threadpool_limits

from unpaired_rosetta.ablation.initializations import ClusterMatching
from unpaired_rosetta.ablation.readouts import relative_representation_readout
from unpaired_rosetta.ablation.refinements import BATCH_SIZE, NUM_CENTERS, cluster_step, neighbor_step
from unpaired_rosetta.determinism import KMEANS_THREADS
from unpaired_rosetta.linalg import center_and_normalize, polar
from unpaired_rosetta.randomness import Randomness

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

OFFICIAL = Path(__file__).parent / "official"
pytestmark = [
    pytest.mark.official,
    pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/mini_vec2vec_components/clone.sh"),
]
SEED = 3


@pytest.fixture(scope="module")
def cells() -> list[str]:
    notebook = nbformat.read(OFFICIAL / "linear_vec2vec.ipynb", as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def definitions(
    cells: list[str],
    **names: object,
) -> dict:
    """The imports and functions of the notebook, without downloads or training, plus ``names``, which replace the
    notebook's own."""
    namespace = {}
    for source in cells:
        module = ast.parse(source)
        module.body = [
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom))
            and "datasets" not in ast.unparse(node)
            and "huggingface_hub" not in ast.unparse(node)
        ]
        exec(compile(module, "<notebook>", "exec"), namespace)
    namespace.update(names)
    return namespace


def cell(
    cells: list[str],
    marker: str,
) -> str:
    return next(source for source in cells if marker in source)


def problem(
    num_samples: int = 700,
    dim: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clustered samples and a rotated copy in another order, centered and normalized as in the notebook."""
    generator = torch.Generator().manual_seed(0)
    centers = 3 * torch.randn(25, dim, generator=generator)
    labels = torch.randint(0, 25, (num_samples,), generator=generator)
    samples_x = centers[labels] + 0.4 * torch.randn(num_samples, dim, generator=generator)
    rotation = polar(torch.randn(dim, dim, generator=generator))
    samples_y = (samples_x @ rotation)[torch.randperm(num_samples, generator=generator)]
    preprocess = lambda samples: center_and_normalize(samples, samples.mean(dim=0, keepdim=True))  # noqa: E731
    return preprocess(samples_x), preprocess(samples_y)


def seeded_kmeans(
    n_clusters: int,
    init: str | torch.Tensor | np.ndarray = "k-means++",
) -> SingleThreadKMeans:
    """``KMeans`` with a seed from the global NumPy generator, as in our code."""
    if isinstance(init, torch.Tensor):
        init = init.numpy()
    return SingleThreadKMeans(n_clusters=n_clusters, init=init, random_state=int(np.random.randint(0, 2**31 - 1)))


class SingleThreadKMeans(sklearn.cluster.KMeans):
    def fit(
        self,
        samples: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> Self:
        with threadpool_limits(KMEANS_THREADS):
            return super().fit(samples, *args, **kwargs)


def seeded_two_opt(
    kernel1: np.ndarray,
    kernel2: np.ndarray,
    method: str,
    options: dict[str, object],
) -> OptimizeResult:
    """The scipy 2-opt from the same random permutation as our code. The notebook's ``P0`` is not an option of the
    scipy 2-opt and is ignored."""
    seed = torch.randint(0, 2**32 - 1, (1,)).item()
    start = np.random.RandomState(seed).permutation(kernel1.shape[0])
    options = {"maximize": options["maximize"], "partial_guess": np.column_stack([np.arange(len(start)), start])}
    return quadratic_assignment(
        np.asarray(kernel1, dtype=np.float64), np.asarray(kernel2, dtype=np.float64), method=method, options=options
    )


def test_two_opt_cluster_matching_is_the_notebooks(
    cells: list[str],
) -> None:
    samples_x, samples_y = problem()
    matched = ClusterMatching(num_clusters=20, solver="2-opt", num_restarts=3, batch_size=500)(
        samples_x, samples_y, Randomness(SEED)
    )

    namespace = definitions(cells, KMeans=seeded_kmeans, quadratic_assignment=seeded_two_opt)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    for centers_x, matched_y in matched:
        official_x, official_y = namespace["aligned_centroids"](
            samples_x, samples_y, n_runs=30, n_clusters=20, method="2opt", subsample=500
        )
        torch.testing.assert_close(official_x, centers_x)
        torch.testing.assert_close(official_y, matched_y)


def test_relative_representation_readout_is_the_notebooks_for_unit_anchors(
    cells: list[str],
) -> None:
    samples_x, samples_y = problem()
    matched = ClusterMatching(num_clusters=20, solver="2-opt", num_restarts=2, batch_size=500)(
        samples_x, samples_y, Randomness(SEED)
    )
    anchors_x = torch.nn.functional.normalize(torch.cat([a for a, _ in matched]), dim=-1)
    anchors_y = torch.nn.functional.normalize(torch.cat([b for _, b in matched]), dim=-1)

    namespace = definitions(cells, X_train=samples_x, Y_train=samples_y, all_centers1=anchors_x, all_centers2=anchors_y)
    for marker in (
        "sim1 = cos_sim_matrix",
        "top_similar = sim_similarity",
        "Y_matched = Y_train",
        "W = train_orthogonal_linear",
    ):
        exec(cell(cells, marker), namespace)
    torch.testing.assert_close(
        relative_representation_readout(samples_x, samples_y, anchors_x, anchors_y), namespace["W"], atol=1e-5, rtol=0
    )


def test_refine1_step_is_the_notebooks(
    cells: list[str],
) -> None:
    samples_x, samples_y = problem()
    weight = polar(torch.randn(6, 6, generator=torch.Generator().manual_seed(1)))
    ours = polar(neighbor_step(samples_x, samples_y, weight, Randomness(SEED)))

    source = cell(cells, "sample_points = X_train").replace("n_iters = 100", "n_iters = 1")
    source = source.replace("[:1000]", f"[:{BATCH_SIZE}]").replace("W = 0.5 * W + 0.5 * W_new", "")
    namespace = definitions(
        cells,
        X_train=samples_x,
        Y_train=samples_y,
        W=weight,
        X_eval=samples_x[:5],
        Y_eval=samples_y[:5],
        print=lambda *args: None,
    )
    torch.manual_seed(SEED)
    exec(source, namespace)
    torch.testing.assert_close(ours, namespace["W_new"], atol=1e-5, rtol=0)


def test_refine2_step_is_the_notebooks(
    cells: list[str],
) -> None:
    samples_x, samples_y = problem()
    weight = polar(torch.randn(6, 6, generator=torch.Generator().manual_seed(1)))
    randomness = Randomness(SEED)
    ours = polar(cluster_step(samples_x, samples_y, weight, randomness))

    # Our step clusters the rows in random order and the notebook in the given order, so the notebook gets them
    # shuffled.
    replay = Randomness(SEED)
    shuffled_x, shuffled_y = (
        samples_x[replay.permutation(samples_x.shape[0])],
        samples_y[replay.permutation(samples_y.shape[0])],
    )
    source = cell(cells, "kmeans1 = KMeans(n_clusters=500)").replace("W = 0.5 * W + 0.5 * W_new", "")
    source = source.replace("n_clusters=500", f"n_clusters={NUM_CENTERS}")
    # The notebook passes torch tensors to KMeans, which scikit-learn converts to float64. We cluster in float32 like
    # the runs behind Table 4, so the notebook gets float32 arrays here.
    namespace = definitions(
        cells,
        X_train=shuffled_x.numpy(),
        Y_train=shuffled_y.numpy(),
        W=weight,
        X_eval=samples_x[:5],
        Y_eval=samples_y[:5],
        print=lambda *args: None,
        KMeans=seeded_kmeans,
    )
    np.random.seed(SEED)
    exec(source, namespace)
    torch.testing.assert_close(ours, namespace["W_new"], atol=1e-5, rtol=0)
