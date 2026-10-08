"""Compare our STRUCTURE baseline with the official code (github.com/mlbio-epfl/STRUCTURE), cloned by ``clone.sh``.

The tests run the ``train`` method (one epoch) of the official trainer from its source, without the heavy logging
dependencies. With all known pairs in one batch, an epoch is one step, so it can be compared with ours step by step.
"""

import ast
import importlib
import sys
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from unpaired_rosetta.baselines import structure
from unpaired_rosetta.baselines.structure import LOSS_SETTINGS, STRUCTURE, learning_rate_factor, training_step
from unpaired_rosetta.linalg import center_and_normalize
from unpaired_rosetta.randomness import Randomness

OFFICIAL = Path(__file__).parent / "official"
pytestmark = [
    pytest.mark.official,
    pytest.mark.skipif(not (OFFICIAL / "src").exists(), reason="run tests/structure/clone.sh"),
]


def import_official(
    module: str,
) -> types.ModuleType:
    """Import a module of the clone under its own ``src`` package and leave the ``src`` of the submodule untouched."""
    structure.load_clip_loss()  # adds a stub for loguru if it is missing
    for name in ("cca_zoo", "cca_zoo.linear"):  # only the CCA baseline of the package imports these
        sys.modules.setdefault(name, types.SimpleNamespace(CCA=type("CCA", (), {})))
    saved = {name: sys.modules.pop(name) for name in list(sys.modules) if name == "src" or name.startswith("src.")}
    sys.path.insert(0, str(OFFICIAL))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(OFFICIAL))
        for name in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture(scope="module")
def official() -> types.SimpleNamespace:
    loss = import_official("src.loss.clip_loss")
    layers = import_official("src.alignment.linear_alignment_layer")
    utils = import_official("src.core.src.utils.utils")
    return types.SimpleNamespace(
        CLIPLoss=loss.CLIPLoss, LinearAlignmentLayer=layers.LinearAlignmentLayer, clip_gradients=utils.clip_gradients
    )


def official_train_function(
    clip_gradients: Callable[[nn.Module, float], None],
) -> Callable[..., None]:
    """``AlignmentTrainer.train`` of the official trainer, compiled from its source."""
    source = (OFFICIAL / "src" / "trainers" / "alignment_trainer.py").read_text()
    trainer = next(
        node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "AlignmentTrainer"
    )
    function = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "train")

    class MeanMetric:  # replaces torchmetrics.MeanMetric, which is only used for logging
        def __init__(
            self,
        ) -> None:
            self.values: list[float] = []

        def to(
            self,
            device: str | torch.device,
        ) -> "MeanMetric":
            return self

        def update(
            self,
            value: torch.Tensor,
            weight: int,
        ) -> None:
            self.values.append(float(value.detach()))

        def compute(
            self,
        ) -> torch.Tensor:
            return torch.tensor(sum(self.values) / len(self.values))

    namespace = {
        "torch": torch,
        "np": np,
        "Optional": __import__("typing").Optional,
        "clip_gradients": clip_gradients,
        "torchmetrics": types.SimpleNamespace(MeanMetric=MeanMetric),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "alignment_trainer.py", "exec"), namespace)
    return namespace["train"]


def official_trainer(
    loss_function: nn.Module,
    batch_size: int,
) -> types.SimpleNamespace:
    config = {
        "training": {
            "mixup_alpha": 0.0,
            "fixed_structure": False,
            "clip_grad": structure.GRADIENT_CLIP,
            "embedding_visualization": 10**9,
        }
    }
    return types.SimpleNamespace(
        config=config, loss=loss_function, train_batch_size=batch_size, device="cpu", wandb_logging=False
    )


def clip_norm_of_each_map(
    model: nn.Module,
    clip: float,
) -> None:
    """The gradient clipping of our loop, ``clip_grad_norm_`` over all parameters of one map."""
    nn.utils.clip_grad_norm_(model.parameters(), clip)


@pytest.fixture
def data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    samples_x, samples_y = torch.randn(40, 12, generator=generator), torch.randn(40, 16, generator=generator)
    paired_x, paired_y = torch.randn(6, 12, generator=generator), torch.randn(6, 16, generator=generator)
    return samples_x, samples_y, paired_x, paired_y


def test_loss_settings_are_the_official_linear_configuration(
    official: types.SimpleNamespace,
) -> None:
    class IncludeLoader(yaml.SafeLoader):
        pass

    IncludeLoader.add_constructor("!include", lambda loader, node: None)
    default = yaml.load((OFFICIAL / "configs" / "default.yaml").read_text(), IncludeLoader)["training"]
    linear = yaml.load((OFFICIAL / "configs" / "losses_lin" / "clip_structure_1.yaml").read_text(), IncludeLoader)
    linear = linear["overrides"]["training"]
    loss = official.CLIPLoss(**linear["clip_loss"])
    assert {key: value for key, value in LOSS_SETTINGS.items() if key in linear["clip_loss"]} == linear["clip_loss"]
    assert LOSS_SETTINGS["structure_centering"] == loss.structure_centering.name
    assert LOSS_SETTINGS["structure_distance"] == loss.structure_distance.name
    assert STRUCTURE().dim_alignment == linear["alignment_layer_kwargs"]["dim_alignment"]
    assert STRUCTURE().batch_size == default["batch_size"]
    assert structure.LEARNING_RATE == default["learning_rate"]
    assert structure.WEIGHT_DECAY == default["optimizer_kwargs"]["weight_decay"]
    assert list(structure.BETAS) == default["optimizer_kwargs"]["betas"]
    assert structure.GRADIENT_CLIP == default["clip_grad"]


def test_loss_equals_official(
    official: types.SimpleNamespace,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    samples_x, samples_y, paired_x, paired_y = data
    ours, theirs = structure.load_clip_loss()(**LOSS_SETTINGS), official.CLIPLoss(**LOSS_SETTINGS)
    ours.warmup_steps = theirs.warmup_steps = 2  # reach the full structure weight within the test
    map_x, map_y = nn.Linear(12, 8), nn.Linear(16, 8)
    for _ in range(3):
        ours.step(), theirs.step()
        values = []
        for loss in (ours, theirs):
            map_x.zero_grad(), map_y.zero_grad()
            value = loss(
                map_x(paired_x),
                map_y(paired_y),
                paired_x,
                paired_y,
                (samples_x, map_x(samples_x)),
                (samples_y, map_y(samples_y)),
            )["overall_loss"]
            value.backward()
            values.append((value.detach(), map_x.weight.grad.clone(), map_y.weight.grad.clone()))
        for mine, official_value in zip(*values):
            assert torch.equal(mine, official_value)


def test_initialization_equals_official_alignment_layers(
    official: types.SimpleNamespace,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    samples_x, samples_y, _, _ = data
    ours = STRUCTURE(dim_alignment=8).fit(samples_x, samples_y, None, None, Randomness(3))
    torch.manual_seed(3)  # in a fresh run the maps are the first draws of the run stream
    layer_x, layer_y = official.LinearAlignmentLayer(12, 8), official.LinearAlignmentLayer(16, 8)
    assert torch.equal(ours.weight_x, layer_x.linear_mapping.weight)
    assert torch.equal(ours.bias_y, layer_y.linear_mapping.bias)
    queries = center_and_normalize(samples_x, samples_x.mean(0, keepdim=True))
    expected = nn.functional.normalize(layer_x(queries), dim=-1)
    assert torch.allclose(ours.embed_x(samples_x), expected, atol=1e-6)


def run_official_epochs(
    official: types.SimpleNamespace,
    clip_gradients: Callable[[nn.Module, float], None],
    maps: list[nn.Linear],
    paired: tuple[torch.Tensor, torch.Tensor],
    unpaired: tuple[torch.Tensor, torch.Tensor],
    num_steps: int,
    batch_size: int,
    seed: int,
) -> tuple[list[nn.Module], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Run official epochs (one step each) on copies of ``maps``. Return the maps and the batches of each epoch."""
    layers = []
    for linear in maps:
        layer = official.LinearAlignmentLayer(linear.in_features, linear.out_features)
        layer.linear_mapping.load_state_dict(linear.state_dict())
        layers.append(layer)
    parameters = list(layers[0].parameters()) + list(layers[1].parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=structure.LEARNING_RATE, weight_decay=structure.WEIGHT_DECAY, betas=structure.BETAS
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, num_steps))
    trainer = official_trainer(official.CLIPLoss(**LOSS_SETTINGS), batch_size)
    train = official_train_function(clip_gradients)
    batches = []
    for epoch in range(num_steps):
        torch.manual_seed(seed + epoch)  # replay the permutations of the epoch to get its batches
        pairs, _ = torch.randperm(paired[0].shape[0]), torch.randperm(paired[0].shape[0])
        order_x, order_y = torch.randperm(unpaired[0].shape[0]), torch.randperm(unpaired[1].shape[0])
        batches.append((paired[0][pairs], paired[1][pairs], unpaired[0][order_x], unpaired[1][order_y]))
        torch.manual_seed(seed + epoch)
        train(
            trainer,
            epoch + 1,
            0,
            paired[0],
            paired[1],
            layers[0],
            layers[1],
            optimizer,
            scheduler,
            unpaired[0],
            unpaired[1],
        )
    return layers, batches


def test_training_steps_equal_official_epochs(
    official: types.SimpleNamespace,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Three steps from the same weights on the same batches give the same parameters as the official loop.

    The gradient clipping is the one difference, so the official loop clips like ours here. We clip the norm of each
    map, and the official code clips the norm of each parameter tensor.
    """
    samples_x, samples_y, paired_x, paired_y = data
    torch.manual_seed(1)
    maps = [nn.Linear(12, 8), nn.Linear(16, 8)]
    layers, batches = run_official_epochs(
        official, clip_norm_of_each_map, maps, (paired_x, paired_y), (samples_x, samples_y), 3, 64, seed=10
    )
    loss_function = structure.load_clip_loss()(**LOSS_SETTINGS)
    parameters = list(maps[0].parameters()) + list(maps[1].parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=structure.LEARNING_RATE, weight_decay=structure.WEIGHT_DECAY, betas=structure.BETAS
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, 3))
    for batch in batches:
        training_step(maps[0], maps[1], loss_function, optimizer, scheduler, *batch)
    for linear, layer in zip(maps, layers):
        assert torch.equal(linear.weight, layer.linear_mapping.weight)
        assert torch.equal(linear.bias, layer.linear_mapping.bias)


def test_fit_and_similarity_equal_official_training(
    official: types.SimpleNamespace,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Our fit equals official epochs from the same start on the same preprocessed data, up to summation order.

    All samples are in one batch, so there is no sampling. Only the order of the permuted batches differs.
    """
    samples_x, samples_y, paired_x, paired_y = data
    ours = STRUCTURE(dim_alignment=8, num_steps=3, batch_size=64).fit(
        samples_x, samples_y, paired_x, paired_y, Randomness(5)
    )

    mean_x, mean_y = samples_x.mean(0, keepdim=True), samples_y.mean(0, keepdim=True)
    torch.manual_seed(5)
    maps = [nn.Linear(12, 8), nn.Linear(16, 8)]
    paired = (center_and_normalize(paired_x, mean_x), center_and_normalize(paired_y, mean_y))
    unpaired = (center_and_normalize(samples_x, mean_x), center_and_normalize(samples_y, mean_y))
    layers, _ = run_official_epochs(official, clip_norm_of_each_map, maps, paired, unpaired, 3, 64, seed=20)

    assert torch.allclose(ours.weight_x, layers[0].linear_mapping.weight, atol=1e-6)
    assert torch.allclose(ours.weight_y, layers[1].linear_mapping.weight, atol=1e-6)
    validation_x, validation_y = torch.randn(9, 12), torch.randn(9, 16)
    embedded_x = nn.functional.normalize(layers[0](center_and_normalize(validation_x, mean_x)), dim=-1)
    embedded_y = nn.functional.normalize(layers[1](center_and_normalize(validation_y, mean_y)), dim=-1)
    assert torch.allclose(ours.similarity(validation_x, validation_y), embedded_x @ embedded_y.T, atol=1e-5)
