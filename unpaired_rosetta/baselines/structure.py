"""STRUCTURE (Groeger et al., 2026): one linear map per modality into a shared space.

The maps are trained with a CLIP loss on the known pairs and a regularizer that keeps the neighborhood structure of
the unpaired samples. The loss is ``CLIPLoss`` from ``external/STRUCTURE`` with the settings of
``configs/losses_lin/clip_structure_1.yaml``. The training loop is our adaptation to the few-pair setting. Each step
uses all known pairs (at most ``batch_size``) and ``batch_size`` random unpaired samples per modality. Without known
pairs the maps stay random (chance level).
"""

import importlib.util
import logging
import math
import sys
import types
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.functional import normalize

from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.linalg import center_and_normalize
from unpaired_rosetta.randomness import Randomness

STRUCTURE_ROOT = Path(__file__).resolve().parents[2] / "external" / "STRUCTURE"

# Settings of the linear STRUCTURE configurations.
LOSS_SETTINGS = dict(
    temperature=0.05,
    normalize_latents=True,
    warmup_steps=1000,
    structure_lambda=10.0,
    structure_levels=1,
    structure_weighting="none",
    structure_margin=0.0,
    structure_centering="mean",
    structure_distance="cosine",
)
LEARNING_RATE = 1e-3  # AdamW settings of STRUCTURE's default configuration
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.95)
LEARNING_RATE_WARMUP_STEPS = 100
GRADIENT_CLIP = 1.0
BATCH_SEED = 0  # batches use their own generator with this seed in every run


def load_clip_loss(
    root: Path = STRUCTURE_ROOT,
) -> type:
    """STRUCTURE's ``CLIPLoss``. Its utilities import loguru only for logging, so a stub is used if it is missing."""
    if "loguru" not in sys.modules and importlib.util.find_spec("loguru") is None:
        sys.modules["loguru"] = types.SimpleNamespace(logger=logging.getLogger("STRUCTURE"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.loss.clip_loss import CLIPLoss

    return CLIPLoss


def learning_rate_factor(
    step: int,
    num_steps: int,
) -> float:
    """Linear warm-up over the first 100 steps, then cosine decay to zero at ``num_steps``."""
    if step < LEARNING_RATE_WARMUP_STEPS:
        return (step + 1) / LEARNING_RATE_WARMUP_STEPS
    progress = (step - LEARNING_RATE_WARMUP_STEPS) / max(1, num_steps - LEARNING_RATE_WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


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
    map_x: nn.Module,
    map_y: nn.Module,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    paired_x: Tensor,
    paired_y: Tensor,
    unpaired_x: Tensor,
    unpaired_y: Tensor,
) -> float:
    """One training step, in the same order as STRUCTURE's trainer."""
    loss_function.step()
    optimizer.zero_grad()
    loss = loss_function(
        image_embeddings_aligned=map_x(paired_x),
        text_embeddings_aligned=map_y(paired_y),
        image_embeddings_original=paired_x,
        text_embeddings_original=paired_y,
        add_image_features=(unpaired_x, map_x(unpaired_x)),
        add_text_features=(unpaired_y, map_y(unpaired_y)),
    )["overall_loss"]
    loss.backward()
    nn.utils.clip_grad_norm_(map_x.parameters(), GRADIENT_CLIP)
    nn.utils.clip_grad_norm_(map_y.parameters(), GRADIENT_CLIP)
    optimizer.step()
    scheduler.step()
    return loss.item()


class STRUCTURE:
    def __init__(
        self,
        dim_alignment: int = 512,
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
    ) -> "STRUCTURE":
        samples_x, samples_y = samples_x.float().to(self.device), samples_y.float().to(self.device)
        self.mean_x = samples_x.mean(dim=0, keepdim=True)
        self.mean_y = samples_y.mean(dim=0, keepdim=True)
        samples_x = center_and_normalize(samples_x, self.mean_x)
        samples_y = center_and_normalize(samples_y, self.mean_y)
        with randomness.global_generators():  # initialized from the run's random stream
            map_x = nn.Linear(samples_x.shape[1], self.dim_alignment).to(self.device)
            map_y = nn.Linear(samples_y.shape[1], self.dim_alignment).to(self.device)
        if paired_x is not None and paired_x.shape[0] > 0:
            paired_x = center_and_normalize(paired_x.float().to(self.device), self.mean_x)
            paired_y = center_and_normalize(paired_y.float().to(self.device), self.mean_y)
            self.train(map_x, map_y, samples_x, samples_y, paired_x, paired_y)
        self.weight_x, self.bias_x = map_x.weight.detach().clone(), map_x.bias.detach().clone()
        self.weight_y, self.bias_y = map_y.weight.detach().clone(), map_y.bias.detach().clone()
        return self

    def train(
        self,
        map_x: nn.Linear,
        map_y: nn.Linear,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor,
        paired_y: Tensor,
    ) -> None:
        loss_function = load_clip_loss()(**LOSS_SETTINGS)
        parameters = list(map_x.parameters()) + list(map_y.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=BETAS)
        schedule = lambda step: learning_rate_factor(step, self.num_steps)  # noqa: E731
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        generator = torch.Generator().manual_seed(BATCH_SEED)
        with torch.enable_grad():
            for _ in range(self.num_steps):
                batch_paired_x, batch_paired_y = sample_rows([paired_x, paired_y], self.batch_size, generator)
                (batch_x,) = sample_rows([samples_x], self.batch_size, generator)
                (batch_y,) = sample_rows([samples_y], self.batch_size, generator)
                training_step(
                    map_x, map_y, loss_function, optimizer, scheduler, batch_paired_x, batch_paired_y, batch_x, batch_y
                )

    def embed_x(
        self,
        samples_x: Tensor,
    ) -> Tensor:
        """Map samples of X into the shared space, with unit length."""
        samples_x = center_and_normalize(samples_x.float().to(self.device), self.mean_x)
        return normalize(samples_x @ self.weight_x.T + self.bias_x, dim=-1)

    def embed_y(
        self,
        samples_y: Tensor,
    ) -> Tensor:
        samples_y = center_and_normalize(samples_y.float().to(self.device), self.mean_y)
        return normalize(samples_y @ self.weight_y.T + self.bias_y, dim=-1)

    @torch.inference_mode()
    def similarity(
        self,
        queries_x: Tensor,
        keys_y: Tensor,
    ) -> Tensor:
        """Cosine similarity in the shared space."""
        return self.embed_x(queries_x) @ self.embed_y(keys_y).T
