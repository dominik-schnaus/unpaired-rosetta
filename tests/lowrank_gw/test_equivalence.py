"""Compare the low-rank Gromov-Wasserstein ablation (Table 2) with the official code of Scetbon, Peyre and Cuturi
(2022, github.com/meyerscetbon/LinearGromov) and of SCOT (Demetci et al., 2020, github.com/rsinghlab/SCOT).

Equal to the official code and tested here:
* the Gromov-Wasserstein loss of a dense plan and of a low-rank plan ``Q diag(1/g) R^T``,
* the ``trivial`` initialization, up to single precision, because our port builds it through POT,
* the geodesic costs of the single-cell problems (the k-NN graph distances of SCOT).

Specific to our ablation and without an official counterpart (see ``unpaired_rosetta/ablation/lowrank_gw.py``):
* the ``random`` start draws the same construction (``|N(0, 1)| + 1``, rows scaled to the marginals) from the torch
  generator of POT instead of the global NumPy generator,
* the ``lower bound`` start runs the low-rank Sinkhorn of POT on the eccentricities instead of the official
  ``Lin_LOT_MD``, and the ``k-means`` start uses hard clusters instead of Sinkhorn soft assignments,
* the mirror descent (step size, projection, stopping rule). The module docstring lists every difference.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from tests.official_code import official_code
from unpaired_rosetta.ablation import lowrank_gw
from unpaired_rosetta.randomness import Randomness

if TYPE_CHECKING:
    from types import ModuleType

OFFICIAL = Path(__file__).parent / "official"
pytestmark = [
    pytest.mark.official,
    pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/lowrank_gw/clone.sh"),
]


@pytest.fixture(scope="module")
def official() -> ModuleType:
    with official_code(OFFICIAL / "LinearGromov"):
        import FastGromovWass

    return FastGromovWass


@pytest.fixture(scope="module")
def scot() -> ModuleType:
    with official_code(OFFICIAL / "SCOT" / "src"):
        import scotv1

    return scotv1


def problem(
    num_x: int = 60,
    num_y: int = 50,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    samples_x = torch.randn(num_x, 4, generator=generator, dtype=torch.float64)
    samples_y = torch.randn(num_y, 6, generator=generator, dtype=torch.float64)
    return (
        samples_x,
        samples_y,
        lowrank_gw.squared_euclidean_cost(samples_x),
        lowrank_gw.squared_euclidean_cost(samples_y),
    )


def test_gw_loss_of_the_independent_plan(
    official: ModuleType,
) -> None:
    _, _, cost_x, cost_y = problem()
    a, b = lowrank_gw.uniform(cost_x.shape[0]), lowrank_gw.uniform(cost_y.shape[0])
    expected = official.GW_init_cubic(cost_x.numpy(), cost_y.numpy(), a.numpy(), b.numpy())
    assert lowrank_gw.gw_loss(cost_x, cost_y, torch.outer(a, b)) == pytest.approx(expected, rel=1e-12)


def test_gw_loss_of_a_low_rank_plan(
    official: ModuleType,
) -> None:
    """The loss equals the official loss ``c + <-4 C_X Q diag(1/g) R^T C_Y R diag(1/g), Q> / 2`` of ``Quad_LGW_MD``."""
    samples_x, samples_y, cost_x, cost_y = problem()
    factors = lowrank_gw.initial_factors(samples_x, samples_y, 5, "k-means", Randomness(0))
    q, r, g = (factor.numpy() for factor in factors)
    a, b = q.sum(axis=1), r.sum(axis=1)
    constant = (cost_x.numpy() ** 2 @ a) @ a + (cost_y.numpy() ** 2 @ b) @ b
    factor_x, factor_y, _ = official.update_Quad_cost_GW(cost_x.numpy(), cost_y.numpy(), q, r, g)
    expected = constant + np.trace(q.T @ (factor_x @ (factor_y @ r) / g)) / 2
    assert lowrank_gw.gw_loss(cost_x, cost_y, lowrank_gw.dense_plan(factors)) == pytest.approx(expected, rel=1e-10)


def test_trivial_initialization(
    official: ModuleType,
) -> None:
    """The trivial start equals the official one.

    With ``max_iter=0``, ``Quad_LGW_MD`` returns only its starting coupling.
    """
    samples_x, samples_y, cost_x, cost_y = problem()
    a, b = lowrank_gw.uniform(cost_x.shape[0]).numpy(), lowrank_gw.uniform(cost_y.shape[0]).numpy()
    *_, couplings = official.Quad_LGW_MD(
        samples_x.numpy(),
        samples_y.numpy(),
        a,
        b,
        7,
        (cost_x.numpy(), cost_y.numpy()),
        max_iter=0,
        C_init=True,
        Init="trivial",
        rescale_cost=False,
    )
    ours = lowrank_gw.initial_factors(samples_x, samples_y, 7, "trivial", Randomness(0))
    for our_factor, official_factor in zip(ours, couplings[0]):
        np.testing.assert_allclose(our_factor.numpy(), official_factor, rtol=1e-6)


def test_geodesic_cost(
    scot: ModuleType,
) -> None:
    samples = torch.randn(120, 8, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    reference = scot.SCOT(samples.numpy(), samples.numpy())
    reference.construct_graph(50, mode="connectivity", metric="correlation")
    expected, _ = reference.init_distances()
    np.testing.assert_array_equal(lowrank_gw.geodesic_cost(samples).numpy(), expected)
