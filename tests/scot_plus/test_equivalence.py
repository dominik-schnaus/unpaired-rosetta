"""Compare our SCOT+ port with the official SCOT code (github.com/rsinghlab/SCOT, ``src/scotv1.py``) and the
official SCOT+ solver (``scotplus`` 1.0.2) on small synthetic data.

SCOT fixes the geometry (kNN-graph geodesic distances), the self-tuning grid and criterion, and the barycentric
projection. SCOT+ provides the entropic Gromov-Wasserstein solver that our port calls. We added the orthogonal map,
the random subsets and the ensemble to evaluate on unseen samples, so they have no official counterpart.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch.nn.functional import normalize

from unpaired_rosetta.baselines import scot_plus
from unpaired_rosetta.baselines.scot_plus import (
    EPSILON_GRID,
    NUM_NEIGHBORS_GRID,
    graph_distances,
    gromov_wasserstein_cost,
    self_tuned_coupling,
)

if TYPE_CHECKING:
    from types import ModuleType

OFFICIAL = Path(__file__).parent / "official"
pytestmark = [
    pytest.mark.official,
    pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/scot_plus/clone.sh"),
]


@pytest.fixture(scope="module")
def scotv1() -> ModuleType:
    sys.path.insert(0, str(OFFICIAL / "src"))
    import scotv1

    return scotv1


@pytest.fixture
def domains() -> tuple[torch.Tensor, torch.Tensor]:
    """Two unit-length point clouds of different widths with the same cluster structure."""
    generator = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 4, (80,), generator=generator)

    def cloud(
        width: int,
    ) -> torch.Tensor:
        centers = torch.randn(4, width, generator=generator, dtype=torch.float64)
        noise = torch.randn(80, width, generator=generator, dtype=torch.float64)
        return normalize(centers[labels] + 0.3 * noise, dim=-1)

    return cloud(12), cloud(9)


@pytest.mark.parametrize("num_neighbors", [5, 20])
def test_graph_distances_equal_scot(
    scotv1: ModuleType,
    domains: tuple[torch.Tensor, torch.Tensor],
    num_neighbors: int,
) -> None:
    x, y = domains
    scot = scotv1.SCOT(x.numpy(), y.numpy())
    scot.construct_graph(num_neighbors, mode="connectivity", metric="correlation")
    official_x, official_y = scot.init_distances()
    assert torch.equal(graph_distances(x, num_neighbors), torch.from_numpy(official_x))
    assert torch.equal(graph_distances(y, num_neighbors), torch.from_numpy(official_y))


def test_self_tuning_grid_equals_scot(
    scotv1: ModuleType,
) -> None:
    """``unsupervised_scot`` searches k in linspace(20, 100, 5) and epsilon in logspace(-1, -3, 12) above 250 cells."""
    scot = scotv1.SCOT(np.zeros((300, 3)), np.zeros((300, 3)))
    searched: dict[str, np.ndarray] = {}

    def search_scot(
        ks: np.ndarray,
        es: np.ndarray,
        **kwargs: object,
    ) -> tuple[None, None, float, int, float]:
        searched.update(ks=ks, es=es)
        return None, None, 0.0, 0, 0.0

    scot.search_scot = search_scot
    scot.unsupervised_scot()
    assert NUM_NEIGHBORS_GRID == tuple(searched["ks"].tolist())
    assert EPSILON_GRID == tuple(searched["es"].tolist())


def test_gromov_wasserstein_cost_equals_scot_gw_distance(
    scotv1: ModuleType,
    domains: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Our criterion is the square-loss GW loss ``<|D_X - D_Y|^2, P (x) P>`` that SCOT minimizes over the grid.

    SCOT reads it from POT with the uniform marginals. The entropic coupling meets those marginals up to the tolerance
    of POT.
    """
    import ot

    x, y = domains
    scot = scotv1.SCOT(x.numpy(), y.numpy())
    scot.init_marginals()
    scot.construct_graph(20)
    distances_x, distances_y = scot.init_distances()
    official_distance = scot.find_correspondences(e=2e-2, verbose=False)
    coupling = torch.from_numpy(scot.coupling)
    ours = gromov_wasserstein_cost(torch.from_numpy(distances_x), torch.from_numpy(distances_y), coupling)
    exact = ot.gromov.gwloss(
        *ot.gromov.init_matrix(distances_x, distances_y, scot.coupling.sum(1), scot.coupling.sum(0)), scot.coupling
    )
    assert ours == pytest.approx(exact, rel=1e-12)
    assert ours == pytest.approx(official_distance, rel=1e-4)


def test_barycentric_projection_equals_scot(
    scotv1: ModuleType,
    domains: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Our normalized projection equals the barycentric projection ``P Y / P 1`` of SCOT. Its row scaling drops out."""
    x, y = domains
    scot = scotv1.SCOT(x.numpy(), y.numpy())
    scot.coupling = torch.rand(80, 80, generator=torch.Generator().manual_seed(1), dtype=torch.float64).numpy()
    official, _ = scot.barycentric_projection(XontoY=True)
    ours = normalize(torch.from_numpy(scot.coupling) @ y, dim=-1)
    assert torch.allclose(ours, normalize(torch.from_numpy(official), dim=-1), rtol=0, atol=1e-15)


def test_self_tuned_coupling_is_the_official_scotplus_coupling_of_lowest_gw_loss(
    domains: tuple[torch.Tensor, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each candidate is the unmodified SCOT+ Sinkhorn GW coupling, and we pick the lowest GW loss, as SCOT does."""
    import ot
    from scotplus.solvers import SinkhornSolver
    from scotplus.utils.alignment import compute_graph_distances

    monkeypatch.setattr(scot_plus, "NUM_NEIGHBORS_GRID", (5, 20))
    monkeypatch.setattr(scot_plus, "EPSILON_GRID", (1e-1, 3e-2, 1e-2))
    x, y = domains
    solver = SinkhornSolver(nits_bcd=5, nits_uot=2000, tol_uot=1e-5)
    ours = self_tuned_coupling(x, y, solver)

    candidates = []
    for num_neighbors in (5, 20):
        official_x, official_y = (
            compute_graph_distances(side.numpy(), n_neighbors=num_neighbors, mode="connectivity", metric="correlation")
            for side in (x, y)
        )
        for epsilon in (1e-1, 3e-2, 1e-2):
            coupling = solver.gw(torch.from_numpy(official_x), torch.from_numpy(official_y), eps=epsilon, verbose=False)
            coupling = coupling[0].numpy()
            matrices = ot.gromov.init_matrix(official_x, official_y, coupling.sum(1), coupling.sum(0))
            candidates.append((ot.gromov.gwloss(*matrices, coupling), coupling))
    best = min(candidates, key=lambda candidate: candidate[0])[1]
    assert torch.equal(ours, torch.from_numpy(best))
