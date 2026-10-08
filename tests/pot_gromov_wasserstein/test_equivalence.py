"""Compare the Gromov-Wasserstein initialization of Table 1 with POT (Flamary et al., 2021), the code it calls.

POT is a pinned dependency of the environment (``pot==0.9.7.post1``), so no clone is needed. The initialization
passes torch tensors, which run the torch backend of POT. The reference is the documented NumPy call of POT.
"""

import numpy as np
import pytest
import torch

from unpaired_rosetta.ablation.initializations import GromovWasserstein
from unpaired_rosetta.ablation.sinkhorn import sinkhorn_projection
from unpaired_rosetta.linalg import center_and_normalize
from unpaired_rosetta.randomness import Randomness

ot = pytest.importorskip("ot")
pytestmark = pytest.mark.official


def samples(
    seed: int = 0,
    num_samples: int = 40,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_samples, 5, generator=generator)
    y = torch.randn(num_samples + 3, 7, generator=generator)
    return center_and_normalize(x, x.mean(dim=0, keepdim=True)), center_and_normalize(y, y.mean(dim=0, keepdim=True))


def test_uniform_start_is_pots_gromov_wasserstein() -> None:
    x, y = samples()
    ours = GromovWasserstein("uniform").restart(x, y, Randomness(0))
    xd, yd = x.double().numpy(), y.double().numpy()
    p, q = ot.unif(xd.shape[0]), ot.unif(yd.shape[0])
    plan = ot.gromov.gromov_wasserstein(
        xd @ xd.T, yd @ yd.T, p, q, loss_fun="square_loss", symmetric=True, G0=np.outer(p, q), max_iter=1000
    )
    torch.testing.assert_close(ours, torch.as_tensor(xd.T @ plan @ yd, dtype=torch.float32), atol=1e-6, rtol=1e-5)


def test_random_start_is_a_coupling_that_pot_improves() -> None:
    x, y = samples(1)
    randomness = Randomness(3)
    log_kernel = torch.rand(x.shape[0], y.shape[0], dtype=torch.float64, generator=Randomness(3).torch).log()
    p, q = (
        torch.full((x.shape[0],), 1 / x.shape[0], dtype=torch.float64),
        torch.full((y.shape[0],), 1 / y.shape[0], dtype=torch.float64),
    )
    start = sinkhorn_projection(log_kernel, p, q, num_iterations=1000)
    torch.testing.assert_close(start.sum(dim=1), p)
    torch.testing.assert_close(start.sum(dim=0), q)
    kernel_x, kernel_y = (x @ x.T).double(), (y @ y.T).double()
    loss = lambda plan: ot.gromov.gwloss(*ot.gromov.init_matrix(kernel_x, kernel_y, p, q, "square_loss"), plan).item()  # noqa: E731
    plan = ot.gromov.gromov_wasserstein(
        kernel_x, kernel_y, p, q, loss_fun="square_loss", symmetric=True, G0=start, max_iter=1000
    )
    assert loss(plan) <= loss(start) + 1e-12
    ours = GromovWasserstein("random").restart(x, y, randomness)
    torch.testing.assert_close(ours, (x.double().T @ plan @ y.double()).float())
