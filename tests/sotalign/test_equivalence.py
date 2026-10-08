"""Compare our SOTAlign baseline with the official code (github.com/ExplainableML/SOTAlign), cloned by ``clone.sh``.

At the pinned commit the official repository only releases the KLOT divergence (``src/klot.py``). The teacher and the
training loop follow Algorithm 1 of the paper, so the tests check them against its equations instead.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from timm.optim import Lion
from torch import nn
from torch.nn.functional import logsigmoid, normalize

from unpaired_rosetta.baselines import sotalign
from unpaired_rosetta.baselines.sotalign import SOTAlign, fit_teacher, siglip_loss, training_step
from unpaired_rosetta.linalg import center_and_normalize
from unpaired_rosetta.randomness import Randomness

if TYPE_CHECKING:
    from torch import Tensor

OFFICIAL = Path(__file__).parent / "official"
pytestmark = [
    pytest.mark.official,
    pytest.mark.skipif(not (OFFICIAL / "src" / "klot.py").exists(), reason="run tests/sotalign/clone.sh"),
]


@pytest.fixture(scope="module")
def official_klot() -> nn.Module:
    specification = importlib.util.spec_from_file_location("official_sotalign_klot", OFFICIAL / "src" / "klot.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.KLOT(
        epsilon_student=sotalign.EPSILON_STUDENT,
        epsilon_teacher=sotalign.EPSILON_TEACHER,
        n_iter=sotalign.SINKHORN_ITERATIONS,
    )


@pytest.fixture
def data() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(0)
    samples_x, samples_y = torch.randn(48, 12, generator=generator), torch.randn(48, 16, generator=generator)
    paired_x, paired_y = torch.randn(7, 12, generator=generator), torch.randn(7, 16, generator=generator)
    return samples_x, samples_y, paired_x, paired_y


def test_klot_equals_official(
    official_klot: nn.Module,
) -> None:
    """The KLOT of our loop (the submodule file with our settings) has the official value and gradient."""
    ours = sotalign.load_klot()(
        epsilon_student=sotalign.EPSILON_STUDENT,
        epsilon_teacher=sotalign.EPSILON_TEACHER,
        n_iter=sotalign.SINKHORN_ITERATIONS,
    )
    generator = torch.Generator().manual_seed(1)
    kernel_teacher = torch.rand(20, 20, generator=generator) * 2 - 1
    results = []
    for klot in (ours, official_klot):
        kernel_student = (torch.rand(20, 20, generator=torch.Generator().manual_seed(2)) * 2 - 1).requires_grad_()
        value = klot(kernel_student, kernel_teacher)
        value.backward()
        results.append((value.detach(), kernel_student.grad))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


def test_teacher_is_ridge_cca(
    data: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    """The teacher projections whiten each ridge-regularized covariance and diagonalize the cross-covariance."""
    _, _, paired_x, paired_y = (tensor.double() for tensor in data)
    weight_x, weight_y = fit_teacher(paired_x, paired_y, dim_alignment=1024)
    assert weight_x.shape == (12, 6) and weight_y.shape == (16, 6)  # at most p - 1 components
    centered_x, centered_y = paired_x - paired_x.mean(0), paired_y - paired_y.mean(0)
    covariance_x = centered_x.T @ centered_x / 7 + sotalign.CCA_RIDGE * torch.eye(12, dtype=torch.float64)
    covariance_y = centered_y.T @ centered_y / 7 + sotalign.CCA_RIDGE * torch.eye(16, dtype=torch.float64)
    assert torch.allclose(weight_x.T @ covariance_x @ weight_x, torch.eye(6, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(weight_y.T @ covariance_y @ weight_y, torch.eye(6, dtype=torch.float64), atol=1e-10)
    cross = weight_x.T @ (centered_x.T @ centered_y / 7) @ weight_y
    assert torch.allclose(cross, torch.diag(torch.diagonal(cross)), atol=1e-10)


def test_training_step_uses_official_klot(
    official_klot: nn.Module,
    data: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    """One step of ours equals SigLIP plus alpha times the official KLOT between student and teacher, then Lion."""
    samples_x, samples_y, paired_x, paired_y = data
    teacher_x, teacher_y = fit_teacher(paired_x, paired_y, dim_alignment=1024)
    runs = []
    for step in ("ours", "reference"):
        torch.manual_seed(3)
        student_x, student_y = nn.Linear(12, 6), nn.Linear(16, 6)
        log_scale = nn.Parameter(torch.tensor(math.log(sotalign.LOGIT_SCALE)))
        bias = nn.Parameter(torch.tensor(sotalign.LOGIT_BIAS))
        parameters = list(student_x.parameters()) + list(student_y.parameters()) + [log_scale, bias]
        optimizer = Lion(parameters, lr=sotalign.LEARNING_RATE, weight_decay=sotalign.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        if step == "ours":
            klot = sotalign.load_klot()(
                epsilon_student=sotalign.EPSILON_STUDENT,
                epsilon_teacher=sotalign.EPSILON_TEACHER,
                n_iter=sotalign.SINKHORN_ITERATIONS,
            )
            loss = training_step(
                student_x,
                student_y,
                teacher_x,
                teacher_y,
                log_scale,
                bias,
                klot,
                optimizer,
                scheduler,
                samples_x,
                samples_y,
                paired_x,
                paired_y,
            )
        else:
            kernel_teacher = normalize(samples_x @ teacher_x, dim=-1) @ normalize(samples_y @ teacher_y, dim=-1).T
            kernel_student = normalize(student_x(samples_x), dim=-1) @ normalize(student_y(samples_y), dim=-1).T
            similarity = normalize(student_x(paired_x), dim=-1) @ normalize(student_y(paired_y), dim=-1).T
            logits = log_scale.exp() * similarity + bias
            labels = 2 * torch.eye(7) - 1
            siglip = (
                -logsigmoid(labels * logits).sum() / 7
            )  # summed and divided by the batch size, as in Zhai et al. (2023)
            value = siglip + sotalign.KLOT_WEIGHT * official_klot(kernel_student, kernel_teacher)
            optimizer.zero_grad()
            value.backward()
            optimizer.step()
            loss = value.item()
        runs.append((loss, [parameter.detach().clone() for parameter in parameters]))
    assert runs[0][0] == runs[1][0]
    for ours, reference in zip(runs[0][1], runs[1][1]):
        assert torch.equal(ours, reference)


def test_siglip_loss_matches_its_definition() -> None:
    """-(1/n) sum_ij log sigmoid(z_ij (t s_ij + b)) with z_ij = 1 on the diagonal and -1 elsewhere."""
    similarity = torch.tensor([[0.9, -0.2], [0.1, 0.8]], dtype=torch.float64)
    log_scale, bias = torch.tensor(math.log(20.0), dtype=torch.float64), torch.tensor(-10.0, dtype=torch.float64)
    terms = [(20 * 0.9 - 10), -(20 * -0.2 - 10), -(20 * 0.1 - 10), (20 * 0.8 - 10)]
    expected = -sum(math.log(1 / (1 + math.exp(-term))) for term in terms) / 2
    assert siglip_loss(similarity, log_scale, bias).item() == pytest.approx(expected, rel=1e-12)


def test_fit_initializes_students_with_teacher_and_zero_pairs_from_the_run_stream(
    data: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    samples_x, samples_y, paired_x, paired_y = data
    untrained = SOTAlign(dim_alignment=8, num_steps=0).fit(samples_x, samples_y, paired_x, paired_y, Randomness(4))
    mean_x, mean_y = samples_x.mean(0, keepdim=True), samples_y.mean(0, keepdim=True)
    teacher_x, teacher_y = fit_teacher(
        center_and_normalize(paired_x, mean_x), center_and_normalize(paired_y, mean_y), 8
    )
    assert torch.equal(untrained.weight_x, teacher_x.T) and torch.equal(untrained.weight_y, teacher_y.T)
    assert not untrained.bias_x.any() and not untrained.bias_y.any()

    without_pairs = SOTAlign(dim_alignment=8).fit(samples_x, samples_y, None, None, Randomness(4))
    torch.manual_seed(4)  # in a fresh run the maps are the first draws of the run stream
    assert torch.equal(without_pairs.weight_x, nn.Linear(12, 8).weight)
