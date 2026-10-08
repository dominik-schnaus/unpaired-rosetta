"""SOTAlign (Roschmann et al., 2026): a linear teacher fitted on the known pairs, then linear students per modality.

Stage 1 fits a ridge CCA teacher ``(W_X, W_Y)`` on the known pairs. For a single pair CCA is undefined, so it uses
Procrustes into space Y. Stage 2 starts the students from the teacher and trains them with

    L = SigLIP(cos(f(X_hat), g(Y_hat))) + alpha KLOT(cos(f(X_B), g(Y_B)), cos(X_B W_X, Y_B W_Y)),

where KLOT compares entropic transport plans on random unpaired batches. KLOT is ``src/klot.py`` from
``external/SOTAlign``, the only file released at that commit. The teacher and training loop follow Algorithm 1 of
their paper. Without known pairs the maps stay random (chance level).
"""

import math
import sys
from pathlib import Path

import torch
from timm.optim import Lion
from torch import Tensor, nn
from torch.nn.functional import logsigmoid, normalize

from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.linalg import center_and_normalize, polar
from unpaired_rosetta.randomness import Randomness

SOTALIGN_SOURCE = Path(__file__).resolve().parents[2] / "external" / "SOTAlign" / "src"

CCA_RIDGE = 0.1
EIGENVALUE_FLOOR = 1e-12
LEARNING_RATE = 1e-4  # Lion, as in the SOTAlign paper
WEIGHT_DECAY = 1e-5
KLOT_WEIGHT = 1e-4  # alpha
EPSILON_STUDENT = 0.05  # entropic regularization of the student and teacher plans
EPSILON_TEACHER = 0.01
SINKHORN_ITERATIONS = 100
LOGIT_SCALE = 20.0  # initial SigLIP temperature and bias, both learned
LOGIT_BIAS = -10.0
BATCH_SEED = 0  # batches use their own generator with this seed in every run


def load_klot(
    source: Path = SOTALIGN_SOURCE,
) -> type:
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from klot import KLOT

    return KLOT


def cosine(
    queries: Tensor,
    keys: Tensor,
) -> Tensor:
    return normalize(queries, dim=-1) @ normalize(keys, dim=-1).T


def inverse_square_root(
    covariance: Tensor,
) -> Tensor:
    """``(S + lambda I)^{-1/2}`` for a covariance ``S`` and the CCA ridge ``lambda``."""
    ridge = CCA_RIDGE * torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance + ridge)
    return (eigenvectors * eigenvalues.clamp(min=EIGENVALUE_FLOOR).rsqrt()) @ eigenvectors.T


def fit_teacher(
    paired_x: Tensor,
    paired_y: Tensor,
    dim_alignment: int,
) -> tuple[Tensor, Tensor]:
    """Teacher projections ``(W_X, W_Y)``: ridge CCA with at most ``min(dim_alignment, p - 1)`` components.

    For a single pair it is Procrustes into space Y.
    """
    num_pairs = paired_x.shape[0]
    if num_pairs < 2:
        identity = torch.eye(paired_y.shape[1], device=paired_y.device, dtype=paired_y.dtype)
        return polar(paired_x.T @ paired_y), identity
    centered_x = paired_x - paired_x.mean(0, keepdim=True)
    centered_y = paired_y - paired_y.mean(0, keepdim=True)
    inverse_sqrt_x = inverse_square_root(centered_x.T @ centered_x / num_pairs)
    inverse_sqrt_y = inverse_square_root(centered_y.T @ centered_y / num_pairs)
    correlation = inverse_sqrt_x @ (centered_x.T @ centered_y / num_pairs) @ inverse_sqrt_y
    left, _, right_transposed = torch.linalg.svd(correlation, full_matrices=False)
    dim = min(dim_alignment, paired_x.shape[1], paired_y.shape[1], num_pairs - 1)
    return inverse_sqrt_x @ left[:, :dim], inverse_sqrt_y @ right_transposed[:dim, :].T


def siglip_loss(
    similarity: Tensor,
    log_scale: Tensor,
    bias: Tensor,
) -> Tensor:
    """Sigmoid loss over all pairs of a batch with the diagonal as positives, divided by the batch size."""
    signs = 2.0 * torch.eye(similarity.shape[0], device=similarity.device) - 1.0
    return -logsigmoid(signs * (log_scale.exp() * similarity + bias)).sum() / similarity.shape[0]


def sample_rows(
    tensors: list[Tensor],
    batch_size: int,
    generator: torch.Generator,
) -> list[Tensor]:
    """The same ``batch_size`` random rows of each tensor. Returns all rows without a draw if there are not more."""
    num_rows = tensors[0].shape[0]
    if num_rows <= batch_size:
        return tensors
    indices = torch.randperm(num_rows, generator=generator)[:batch_size].to(tensors[0].device)
    return [tensor[indices] for tensor in tensors]


def training_step(
    student_x: nn.Module,
    student_y: nn.Module,
    teacher_x: Tensor,
    teacher_y: Tensor,
    log_scale: Tensor,
    bias: Tensor,
    klot: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    unpaired_x: Tensor,
    unpaired_y: Tensor,
    paired_x: Tensor,
    paired_y: Tensor,
) -> float:
    """One step of stage 2: SigLIP on the pairs plus alpha KLOT to the teacher on the unpaired batch."""
    with torch.no_grad():
        kernel_teacher = cosine(unpaired_x @ teacher_x, unpaired_y @ teacher_y)
    kernel_student = cosine(student_x(unpaired_x), student_y(unpaired_y))
    loss = siglip_loss(cosine(student_x(paired_x), student_y(paired_y)), log_scale, bias)
    loss = loss + KLOT_WEIGHT * klot(kernel_student, kernel_teacher)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


class SOTAlign:
    def __init__(
        self,
        dim_alignment: int = 1024,
        num_steps: int = 2000,
        batch_size: int = 4096,
        device: str = "cpu",
    ) -> None:
        self.dim_alignment = dim_alignment
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.mean_x = self.mean_y = None
        self.weight_x = self.bias_x = self.weight_y = self.bias_y = None

    @torch_threads()
    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None,
        paired_y: Tensor | None,
        randomness: Randomness,
    ) -> "SOTAlign":
        samples_x, samples_y = samples_x.float().to(self.device), samples_y.float().to(self.device)
        self.mean_x = samples_x.mean(dim=0, keepdim=True)
        self.mean_y = samples_y.mean(dim=0, keepdim=True)
        samples_x = center_and_normalize(samples_x, self.mean_x)
        samples_y = center_and_normalize(samples_y, self.mean_y)
        if paired_x is None or paired_x.shape[0] == 0:
            dim = min(self.dim_alignment, samples_x.shape[1], samples_y.shape[1])
            student_x, student_y = self.students(samples_x.shape[1], samples_y.shape[1], dim, randomness)
        else:
            paired_x = center_and_normalize(paired_x.float().to(self.device), self.mean_x)
            paired_y = center_and_normalize(paired_y.float().to(self.device), self.mean_y)
            teacher_x, teacher_y = fit_teacher(paired_x, paired_y, self.dim_alignment)
            student_x, student_y = self.students(samples_x.shape[1], samples_y.shape[1], teacher_x.shape[1], randomness)
            with torch.no_grad():  # start from the teacher
                student_x.weight.copy_(teacher_x.T)
                student_x.bias.zero_()
                student_y.weight.copy_(teacher_y.T)
                student_y.bias.zero_()
            self.train(student_x, student_y, teacher_x, teacher_y, samples_x, samples_y, paired_x, paired_y)
        self.weight_x, self.bias_x = student_x.weight.detach().clone(), student_x.bias.detach().clone()
        self.weight_y, self.bias_y = student_y.weight.detach().clone(), student_y.bias.detach().clone()
        return self

    def students(
        self,
        dim_x: int,
        dim_y: int,
        dim: int,
        randomness: Randomness,
    ) -> tuple[nn.Linear, nn.Linear]:
        """Two new linear maps, initialized from the run's random stream."""
        with randomness.global_generators():
            return nn.Linear(dim_x, dim).to(self.device), nn.Linear(dim_y, dim).to(self.device)

    def train(
        self,
        student_x: nn.Linear,
        student_y: nn.Linear,
        teacher_x: Tensor,
        teacher_y: Tensor,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor,
        paired_y: Tensor,
    ) -> None:
        klot = load_klot()(epsilon_student=EPSILON_STUDENT, epsilon_teacher=EPSILON_TEACHER, n_iter=SINKHORN_ITERATIONS)
        log_scale = nn.Parameter(torch.tensor(math.log(LOGIT_SCALE), device=self.device))
        bias = nn.Parameter(torch.tensor(LOGIT_BIAS, device=self.device))
        parameters = list(student_x.parameters()) + list(student_y.parameters()) + [log_scale, bias]
        optimizer = Lion(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / max(1, self.num_steps))))
        )
        generator = torch.Generator().manual_seed(BATCH_SEED)
        batch_size = min(self.batch_size, samples_x.shape[0], samples_y.shape[0])
        with torch.enable_grad():
            for _ in range(self.num_steps):
                (batch_x,) = sample_rows([samples_x], batch_size, generator)
                (batch_y,) = sample_rows([samples_y], batch_size, generator)
                batch_paired_x, batch_paired_y = sample_rows([paired_x, paired_y], self.batch_size, generator)
                training_step(
                    student_x,
                    student_y,
                    teacher_x,
                    teacher_y,
                    log_scale,
                    bias,
                    klot,
                    optimizer,
                    scheduler,
                    batch_x,
                    batch_y,
                    batch_paired_x,
                    batch_paired_y,
                )

    def embed_x(
        self,
        samples_x: Tensor,
    ) -> Tensor:
        """Map samples of X into the shared space, without normalizing."""
        return center_and_normalize(samples_x.float().to(self.device), self.mean_x) @ self.weight_x.T + self.bias_x

    def embed_y(
        self,
        samples_y: Tensor,
    ) -> Tensor:
        return center_and_normalize(samples_y.float().to(self.device), self.mean_y) @ self.weight_y.T + self.bias_y

    @torch.inference_mode()
    def similarity(
        self,
        queries_x: Tensor,
        keys_y: Tensor,
    ) -> Tensor:
        """Cosine similarity in the shared space."""
        return cosine(self.embed_x(queries_x), self.embed_y(keys_y))
