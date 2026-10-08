"""Compare our vec2vec with the official code (github.com/rjha18/vec2vec, cloned by ``tests/vec2vec/clone.sh``).

The runs of the paper differ from the official configuration in a few choices (see the module docstring of
``unpaired_rosetta.baselines.vec2vec``). The tests set up the official building blocks with those choices, give both
implementations the same weights and batches, and compare networks, losses and training steps:

* the residual MLPs and discriminators draw the same initial weights and compute the same outputs,
* the reconstruction, cycle and vector space preservation losses agree,
* a least-squares discriminator with labels 1/0 equals the official one (labels 0/1) on the mirrored output 1 - D,
* several steps of our training loop equal the official ``training_loop_`` (losses and parameter updates). For this,
  ours also detaches the cycle input like the official code, and dropout is off, because the two implementations
  call the dropout layers in different orders.
"""

import ast
import copy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.nn import Dropout
from torch.nn.functional import normalize
from tqdm import tqdm

from tests.official_code import official_code
from unpaired_rosetta.baselines.vec2vec import (
    Discriminator,
    LeastSquaresGAN,
    MLPWithResidual,
    Translator,
    Vec2Vec,
    cosine_loss,
    vsp_loss,
)

OFFICIAL = Path(__file__).parent / "official"
pytestmark = [
    pytest.mark.official,
    pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/vec2vec/clone.sh"),
]

DIM_X, DIM_Y, HIDDEN, BATCH = 24, 16, 64, 32


class DummyLogger:
    """Collects what the official training loop logs, one dict per step."""

    steps: list[dict[str, float]]
    current: dict[str, float]

    def __init__(
        self,
    ) -> None:
        self.steps, self.current = [], {}

    def logkv(
        self,
        key: str,
        value: float,
    ) -> None:
        self.current[key] = value

    def dumpkvs(
        self,
        force: bool = False,
    ) -> None:
        self.steps.append(self.current)
        self.current = {}


class Mirrored(torch.nn.Module):
    """``1 - D(x)`` turns our discriminator (real 1, fake 0) into one with the official labels (real 0, fake 1)."""

    def __init__(
        self,
        discriminator: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.discriminator = discriminator

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return 1 - self.discriminator(x)


@pytest.fixture(scope="module")
def official() -> SimpleNamespace:
    """The official modules and ``training_loop_`` from train.py. The other imports of train.py need wandb, vec2text
    and more."""
    with official_code(OFFICIAL):
        from translators.Discriminator import Discriminator as OfficialDiscriminator
        from translators.MLPWithResidual import MLPWithResidual as OfficialMLP
        from translators.TransformTranslator import TransformTranslator
        from utils import gan, train_utils

        source = (OFFICIAL / "train.py").read_text()
        module = ast.parse(source)
        module.body = [node for node in module.body if getattr(node, "name", None) == "training_loop_"]
        namespace = {
            "os": __import__("os"),
            "random": __import__("random"),
            "torch": torch,
            "tqdm": partial(tqdm, disable=True),
            "toml": SimpleNamespace(dump=lambda *args: None),
            "Logger": None,
            "process_batch": None,
            "exit_on_nan": lambda loss: None,
            "rec_loss_fn": train_utils.rec_loss_fn,
            "trans_loss_fn": train_utils.trans_loss_fn,
            "vsp_loss_fn": train_utils.vsp_loss_fn,
            "get_grad_norm": train_utils.get_grad_norm,
        }
        exec(compile(module, str(OFFICIAL / "train.py"), "exec"), namespace)
    return SimpleNamespace(
        MLP=OfficialMLP,
        Discriminator=OfficialDiscriminator,
        TransformTranslator=TransformTranslator,
        LeastSquaresGAN=gan.LeastSquaresGAN,
        train_utils=train_utils,
        training_loop=namespace["training_loop_"],
    )


def disable_dropout(
    module: torch.nn.Module,
) -> None:
    for submodule in module.modules():
        if isinstance(submodule, Dropout):
            submodule.p = 0.0


def official_translator(
    official: SimpleNamespace,
    translator: Translator,
) -> torch.nn.Module:
    """The official translator with our architecture (a depth-3 layer-norm MLP as transform) and our weights."""
    transform = official.MLP(3, HIDDEN, HIDDEN, HIDDEN, "layer")
    model = official.TransformTranslator(
        {"x": DIM_X, "y": DIM_Y}, HIDDEN, HIDDEN, transform, depth=3, normalize_embeddings=True, norm_style="batch"
    )
    model.load_state_dict({official_name(key): value.clone() for key, value in translator.state_dict().items()})
    return model


def official_name(
    name: str,
) -> str:
    """The official name of a parameter or buffer of our translator."""
    prefix, rest = name.split(".", 1)
    prefixes = {
        "input_adapter_x": "in_adapters.x",
        "input_adapter_y": "in_adapters.y",
        "output_adapter_x": "out_adapters.x",
        "output_adapter_y": "out_adapters.y",
        "backbone": "transform",
    }
    return f"{prefixes[prefix]}.{rest}"


def assert_same_update(
    before: torch.Tensor,
    ours: torch.Tensor,
    theirs: torch.Tensor,
) -> None:
    """Check that both parameters moved from ``before`` by the same step, up to the rounding of the stored values."""
    update = (ours - before).abs().max()
    assert update > 0
    assert (ours - theirs).abs().max() <= 1e-3 * update


def batches(
    num_steps: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1)
    return [
        (torch.randn(BATCH, DIM_X, generator=generator), torch.randn(BATCH, DIM_Y, generator=generator))
        for _ in range(num_steps)
    ]


@pytest.mark.parametrize(
    "in_dim, out_dim, norm",
    [(DIM_X, HIDDEN, "batch"), (HIDDEN, HIDDEN, "layer"), (HIDDEN, DIM_Y, "batch")],
    ids=["input adapter", "backbone", "output adapter"],
)
def test_residual_mlp_draws_the_same_weights_and_computes_the_same(
    official: SimpleNamespace,
    in_dim: int,
    out_dim: int,
    norm: str,
) -> None:
    torch.manual_seed(0)
    theirs = official.MLP(3, in_dim, HIDDEN, out_dim, norm)
    torch.manual_seed(0)
    ours = MLPWithResidual(3, in_dim, HIDDEN, out_dim, torch.nn.BatchNorm1d if norm == "batch" else torch.nn.LayerNorm)
    our_state, their_state = ours.state_dict(), theirs.state_dict()
    assert list(our_state) == list(their_state)
    assert all(torch.equal(our_state[key], their_state[key]) for key in our_state)
    x = torch.randn(BATCH, in_dim)
    for mode in (True, False):  # training (dropout, batch statistics) and evaluation
        ours.train(mode), theirs.train(mode)
        torch.manual_seed(2)
        our_output = ours(x)
        torch.manual_seed(2)
        assert torch.equal(our_output, theirs(x))


def test_discriminator_draws_the_official_linear_weights(
    official: SimpleNamespace,
) -> None:
    """The linear layers get the official weights. The official layer norms draw no random numbers."""
    torch.manual_seed(0)
    theirs = official.Discriminator(DIM_X, HIDDEN, depth=5)
    torch.manual_seed(0)
    ours = Discriminator(DIM_X, HIDDEN, depth=5)
    linear = lambda model: [m for m in model.modules() if isinstance(m, torch.nn.Linear)]  # noqa: E731
    for our_layer, their_layer in zip(linear(ours), linear(theirs), strict=True):
        assert torch.equal(our_layer.weight, their_layer.weight) and torch.equal(our_layer.bias, their_layer.bias)


def test_losses_equal_the_official_losses(
    official: SimpleNamespace,
) -> None:
    torch.manual_seed(0)
    samples, translated = normalize(torch.randn(BATCH, DIM_X), dim=1), normalize(torch.randn(BATCH, DIM_X), dim=1)
    logger = DummyLogger()
    reconstruction = official.train_utils.rec_loss_fn({"x": samples}, {"x": translated}, logger)
    torch.testing.assert_close(cosine_loss(translated, samples), reconstruction, rtol=0, atol=1e-7)
    vsp = official.train_utils.vsp_loss_fn({"x": samples}, {"x": {"y": translated}}, logger)
    torch.testing.assert_close(vsp_loss(samples, translated), vsp, rtol=1e-6, atol=1e-7)


def test_least_squares_gan_equals_the_official_one_on_the_mirrored_discriminator(
    official: SimpleNamespace,
) -> None:
    torch.manual_seed(0)
    ours = LeastSquaresGAN(Discriminator(DIM_X, HIDDEN))
    mirrored = Mirrored(copy.deepcopy(ours.discriminator))
    optimizer = torch.optim.Adam(mirrored.parameters(), lr=1e-5, betas=(0.5, 0.999))
    cfg = SimpleNamespace(
        smooth=0.9,
        loss_coefficient_r1_penalty=0.0,
        loss_coefficient_disc=1.0,
        loss_coefficient_gen=1.0,
        max_grad_norm=1000.0,
    )
    import accelerate

    theirs = official.LeastSquaresGAN(
        cfg=cfg,
        generator=torch.nn.Linear(1, 1),
        discriminator=mirrored,
        discriminator_opt=optimizer,
        discriminator_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0),
        accelerator=accelerate.Accelerator(cpu=True),
    )
    before = [parameter.detach().clone() for parameter in ours.discriminator.parameters()]
    for step in range(3):
        real, fake = torch.randn(BATCH, DIM_X), torch.randn(BATCH, DIM_X)
        our_discriminator_loss = ours.update(real, fake)
        our_generator_loss = ours.generator_loss(fake)
        _, their_discriminator_loss, their_generator_loss, *_ = theirs.step(real, fake)
        torch.testing.assert_close(our_discriminator_loss, their_discriminator_loss, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(our_generator_loss, their_generator_loss, rtol=1e-6, atol=1e-7)
    for initial, our_parameter, their_parameter in zip(
        before, ours.discriminator.parameters(), mirrored.parameters(), strict=True
    ):
        assert_same_update(initial, our_parameter.detach(), their_parameter.detach())


def test_training_steps_equal_the_official_training_loop(
    official: SimpleNamespace,
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    ours = Vec2Vec(hidden_dim=HIDDEN, device="cpu")
    ours.build(DIM_X, DIM_Y)
    disable_dropout(ours.translator)
    # The official loop translates back from a detached translation. Ours keeps the gradient of the first translation.
    translator = ours.translator
    translator.translate_xy = lambda samples: Translator.translate_xy(translator, samples.detach())
    translator.translate_yx = lambda samples: Translator.translate_yx(translator, samples.detach())

    their_translator = official_translator(official, ours.translator)
    disable_dropout(their_translator)
    import accelerate

    accelerator = accelerate.Accelerator(cpu=True)
    cfg = SimpleNamespace(
        dataset="fmri",
        normalize_embeddings=True,
        noise_level=0.0,
        sup_emb="x",
        unsup_emb="y",
        loss_coefficient_rec=1.0,
        loss_coefficient_reverse_rec=0.0,
        loss_coefficient_vsp=1.0,
        loss_coefficient_cc_vsp=10.0,
        loss_coefficient_cc_rec=0.0,
        loss_coefficient_cc_trans=10.0,
        loss_coefficient_gen=1.0,
        loss_coefficient_latent_gen=1.0,
        loss_coefficient_similarity_gen=0.0,
        loss_coefficient_disc=1.0,
        loss_coefficient_r1_penalty=0.0,
        smooth=0.9,
        max_grad_norm=1000.0,
    )

    def official_gan(
        our_gan: LeastSquaresGAN,
    ) -> Any:
        mirrored = Mirrored(copy.deepcopy(our_gan.discriminator))
        optimizer = torch.optim.Adam(mirrored.parameters(), lr=1e-5, betas=(0.5, 0.999))
        return official.LeastSquaresGAN(
            cfg=cfg,
            generator=their_translator,
            discriminator=mirrored,
            discriminator_opt=optimizer,
            discriminator_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0),
            accelerator=accelerator,
        )

    gan, sup_gan, latent_gan = official_gan(ours.gan_y), official_gan(ours.gan_x), official_gan(ours.gan_latent)
    unused_gan = SimpleNamespace(discriminator=torch.nn.Linear(1, 1))
    optimizer = torch.optim.Adam(their_translator.parameters(), lr=2e-5, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1, step / 2000))

    data = batches(5)
    before = {name: parameter.detach().clone() for name, parameter in ours.translator.named_parameters()}
    discriminators_before = [
        [parameter.detach().clone() for parameter in our_gan.discriminator.parameters()]
        for our_gan in (ours.gan_y, ours.gan_x, ours.gan_latent)
    ]
    our_losses = [ours.training_step(batch_x, batch_y) for batch_x, batch_y in data]
    logger = DummyLogger()
    official.training_loop(
        save_dir=f"{tmp_path}/",
        accelerator=accelerator,
        gan=gan,
        sup_gan=sup_gan,
        latent_gan=latent_gan,
        similarity_gan=unused_gan,
        translator=their_translator,
        sup_dataloader=[{"x": batch_x} for batch_x, _ in data],
        sup_iter=None,
        unsup_dataloader=[{"y": batch_y} for _, batch_y in data],
        sup_encs={},
        unsup_enc={},
        cfg=cfg,
        gen_optimizers=[optimizer],
        gen_schedulers=[scheduler],
        logger=logger,
    )

    pairs = {  # official logged loss to ours
        "loss": lambda losses: losses["translator"],
        "rec_loss": lambda losses: (losses["reconstruction_x"] + losses["reconstruction_y"]) / 2,
        "vsp_loss": lambda losses: (losses["vsp_x"] + losses["vsp_y"]) / 2,
        "cc_vsp_loss": lambda losses: (losses["cycle_vsp_x"] + losses["cycle_vsp_y"]) / 2,
        "cc_trans_loss": lambda losses: (losses["cycle_x"] + losses["cycle_y"]) / 2,
        "gen_loss": lambda losses: losses["gan_y"],
        "sup_gen_loss": lambda losses: losses["gan_x"],
        "latent_gen_loss": lambda losses: losses["gan_latent"],
        "disc_loss": lambda losses: losses["discriminator_y"],
        "sup_disc_loss": lambda losses: losses["discriminator_x"],
        "latent_disc_loss": lambda losses: losses["discriminator_latent"],
    }
    assert len(logger.steps) == len(our_losses)
    for their_step, our_step in zip(logger.steps, our_losses):
        for key, ours_of in pairs.items():
            assert ours_of(our_step).item() == pytest.approx(their_step[key], rel=1e-5, abs=1e-7), key

    their_parameters = dict(their_translator.named_parameters())
    assert len(their_parameters) == len(before)
    for name, parameter in ours.translator.named_parameters():
        assert_same_update(before[name], parameter.detach(), their_parameters[official_name(name)].detach())
    for our_gan, their_gan, initial in zip(
        (ours.gan_y, ours.gan_x, ours.gan_latent), (gan, sup_gan, latent_gan), discriminators_before
    ):
        for before_step, our_parameter, their_parameter in zip(
            initial, our_gan.discriminator.parameters(), their_gan.discriminator.parameters(), strict=True
        ):
            assert_same_update(before_step, our_parameter.detach(), their_parameter.detach())


def test_similarity_equals_the_official_translation(
    official: SimpleNamespace,
) -> None:
    torch.manual_seed(0)
    ours = Vec2Vec(hidden_dim=HIDDEN, device="cpu")
    ours.build(DIM_X, DIM_Y)
    for batch_x, batch_y in batches(3):  # move the batch norm statistics away from their initial values
        ours.training_step(batch_x, batch_y)
    ours.translator.eval()
    theirs = official_translator(official, ours.translator).eval()
    queries, keys = torch.randn(50, DIM_X), normalize(torch.randn(40, DIM_Y), dim=1)
    with torch.no_grad():
        translated = theirs.translate_embeddings(normalize(queries, dim=1), "x", "y")
    assert torch.equal(ours.similarity(queries, keys), translated @ keys.T)
