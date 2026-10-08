"""vec2vec (Jha et al., 2026): adversarial translation between two embedding spaces through a shared latent space.

Each space has an input adapter into the latent space and an output adapter back, with a shared backbone. The
translator is trained on random batches of the two unpaired sets with least-squares GAN losses, reconstruction and
cycle consistency losses, and vector space preservation (pairwise cosine similarities within a batch are kept).

The networks, loss weights and optimizers follow the official code (github.com/rjha18/vec2vec,
``configs/unsupervised.toml``), with the changes of the runs behind the paper's numbers:

* The backbone has depth 3 with layer norm instead of depth 4 with batch norm.
* The discriminators have no layer norm.
* There is no learning-rate warm-up, and Adam uses its default epsilon.
* The translator trains in float32 for a fixed number of steps.
* The GAN labels real as 1 and fake as 0. The official code swaps them, which only mirrors the discriminator.
* The cycle translation keeps the gradient through the first translation. The official code detaches it.
"""

from functools import partial

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, LayerNorm, Linear, Module, ModuleList, Sequential, SiLU
from torch.nn.functional import cosine_similarity, normalize
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import trange

from unpaired_rosetta.determinism import enable_deterministic_algorithms, torch_threads
from unpaired_rosetta.randomness import Randomness

HIDDEN_DIM = 1024
DEPTH = 3  # depth of each adapter and of the backbone
DISCRIMINATOR_DEPTH = 5
TRANSLATOR_LEARNING_RATE = 2e-5
DISCRIMINATOR_LEARNING_RATE = 1e-5
BETAS = (0.5, 0.999)
WARMUP_STEPS = 2000
MAX_GRADIENT_NORM = 1000.0


def kaiming_initialization(
    module: Module,
) -> None:
    """The official initialization: Kaiming-normal linear weights, zero biases, batch norm scales around one."""
    for submodule in module.modules():
        if isinstance(submodule, Linear):
            torch.nn.init.kaiming_normal_(submodule.weight, a=0, mode="fan_in", nonlinearity="relu")
            submodule.bias.data.fill_(0)
        elif isinstance(submodule, BatchNorm1d):
            torch.nn.init.normal_(submodule.weight, mean=1.0, std=0.02)
            torch.nn.init.normal_(submodule.bias, mean=0.0, std=0.02)
        elif isinstance(submodule, LayerNorm):
            torch.nn.init.constant_(submodule.bias, 0)
            torch.nn.init.constant_(submodule.weight, 1.0)


def add_residual(
    inputs: Tensor,
    outputs: Tensor,
) -> Tensor:
    """Residual connection between layers of different width. The input is zero-padded or truncated."""
    if inputs.shape[1] < outputs.shape[1]:
        padding = torch.zeros(outputs.shape[0], outputs.shape[1] - inputs.shape[1], device=outputs.device)
        inputs = torch.cat([inputs, padding], dim=1)
    elif inputs.shape[1] > outputs.shape[1]:
        inputs = inputs[:, : outputs.shape[1]]
    return outputs + inputs


class MLPWithResidual(Module):
    """The official residual MLP (``translators/MLPWithResidual.py``), built in the same order for the same weights."""

    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        norm_layer: type[Module],
    ) -> None:
        super().__init__()
        self.layers = ModuleList()
        for index in range(depth):
            if index == 0:
                width = out_dim if depth == 1 else hidden_dim
                self.layers.append(Sequential(Linear(in_dim, width), SiLU()))
            elif index < depth - 1:
                self.layers.append(
                    Sequential(Linear(hidden_dim, hidden_dim), SiLU(), norm_layer(hidden_dim), Dropout(p=0.1))
                )
            else:
                self.layers.append(
                    Sequential(Linear(hidden_dim, hidden_dim), Dropout(p=0.1), SiLU(), Linear(hidden_dim, out_dim))
                )
        kaiming_initialization(self)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            x = add_residual(x, layer(x))
        return x


class Discriminator(Module):
    """The official discriminator (``translators/Discriminator.py``) without its layer norms.

    Layer norms draw no random numbers, so the linear weights still match the official ones.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        depth: int = DISCRIMINATOR_DEPTH,
    ) -> None:
        super().__init__()
        layers = [Linear(in_dim, hidden_dim), Dropout(0.0)]
        for _ in range(depth - 2):
            layers += [SiLU(), Linear(hidden_dim, hidden_dim), Dropout(0.0)]
        layers += [SiLU(), Linear(hidden_dim, 1)]
        self.layers = ModuleList([Sequential(*layers)])
        kaiming_initialization(self)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.layers[0](x)


class LeastSquaresGAN:
    """A discriminator with its own optimizer.

    ``update`` trains it on a real and a detached fake batch. ``generator_loss`` is what the translator minimizes.
    """

    def __init__(
        self,
        discriminator: Discriminator,
    ) -> None:
        self.discriminator = discriminator
        self.optimizer = torch.optim.Adam(discriminator.parameters(), lr=DISCRIMINATOR_LEARNING_RATE, betas=BETAS)

    def update(
        self,
        real: Tensor,
        fake: Tensor,
    ) -> Tensor:
        self.optimizer.zero_grad()
        loss = 0.5 * ((self.discriminator(real) - 1.0).pow(2).mean() + self.discriminator(fake).pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), MAX_GRADIENT_NORM)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def generator_loss(
        self,
        fake: Tensor,
    ) -> Tensor:
        return 0.5 * (self.discriminator(fake) - 1.0).pow(2).mean()


def linear_warmup(
    step: int,
    num_steps: int,
) -> float:
    return min(1.0, step / num_steps)


def cosine_loss(
    predictions: Tensor,
    targets: Tensor,
) -> Tensor:
    return 1 - cosine_similarity(predictions, targets, dim=1).mean()


def vsp_loss(
    samples: Tensor,
    translated: Tensor,
) -> Tensor:
    """Vector space preservation loss: the cosine similarities of the translated batch should match the original."""
    samples, translated = normalize(samples, p=2, dim=1), normalize(translated, p=2, dim=1)
    similarities = samples @ samples.T
    translated_similarities = translated @ translated.T
    cross_similarities = samples @ translated.T
    return (similarities - translated_similarities).abs().mean() + (similarities - cross_similarities).abs().mean()


class Translator(Module):
    """Input and output adapters for a shared latent space, with a shared backbone."""

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.input_adapter_x = MLPWithResidual(DEPTH, dim_x, hidden_dim, hidden_dim, BatchNorm1d)
        self.input_adapter_y = MLPWithResidual(DEPTH, dim_y, hidden_dim, hidden_dim, BatchNorm1d)
        self.backbone = MLPWithResidual(DEPTH, hidden_dim, hidden_dim, hidden_dim, LayerNorm)
        self.output_adapter_x = MLPWithResidual(DEPTH, hidden_dim, hidden_dim, dim_x, BatchNorm1d)
        self.output_adapter_y = MLPWithResidual(DEPTH, hidden_dim, hidden_dim, dim_y, BatchNorm1d)

    def translate_xy(
        self,
        samples_x: Tensor,
    ) -> Tensor:
        return normalize(self.output_adapter_y(self.backbone(self.input_adapter_x(samples_x))), p=2, dim=1)

    def translate_yx(
        self,
        samples_y: Tensor,
    ) -> Tensor:
        return normalize(self.output_adapter_x(self.backbone(self.input_adapter_y(samples_y))), p=2, dim=1)


class BatchStream:
    """Batches of a fixed size from a permutation per epoch.

    A batch that crosses an epoch end is filled from a new permutation without the rows already in it, so no batch
    repeats a row.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.permutation = torch.randperm(num_samples, device=device)
        self.cursor = 0

    def next(
        self,
    ) -> Tensor:
        first_take = min(self.batch_size, self.num_samples - self.cursor)
        indices = self.permutation[self.cursor : self.cursor + first_take]
        self.cursor += first_take
        if first_take == self.batch_size:
            return indices
        permutation = torch.randperm(self.num_samples, device=self.device)
        if indices.numel() > 0:
            used = torch.zeros(self.num_samples, dtype=torch.bool, device=self.device)
            used[indices] = True
            permutation = torch.cat((permutation[~used[permutation]], permutation[used[permutation]]))
        self.permutation = permutation
        self.cursor = self.batch_size - first_take
        return torch.cat((indices, permutation[: self.cursor]))


class Vec2Vec:
    """vec2vec aligner that trains the translator for ``num_steps`` steps on batches of ``batch_size`` rows."""

    def __init__(
        self,
        batch_size: int = 256,
        num_steps: int = 100_000,
        hidden_dim: int = HIDDEN_DIM,
        device: str = "cuda",
        verbose: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.verbose = verbose
        self.translator: Translator | None = None

    def build(
        self,
        dim_x: int,
        dim_y: int,
    ) -> None:
        """Networks and their optimizers.

        Built on the CPU in this order, so the initial weights match the runs of the paper.
        """
        self.translator = Translator(dim_x, dim_y, self.hidden_dim)
        self.gan_x = LeastSquaresGAN(Discriminator(dim_x, self.hidden_dim))
        self.gan_y = LeastSquaresGAN(Discriminator(dim_y, self.hidden_dim))
        self.gan_latent = LeastSquaresGAN(Discriminator(self.hidden_dim, self.hidden_dim))
        self.optimizer = torch.optim.Adam(self.translator.parameters(), lr=TRANSLATOR_LEARNING_RATE, betas=BETAS)
        self.scheduler = LambdaLR(self.optimizer, partial(linear_warmup, num_steps=WARMUP_STEPS))
        self.translator.to(self.device)
        for gan in (self.gan_x, self.gan_y, self.gan_latent):
            gan.discriminator.to(self.device)

    @torch_threads()
    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None = None,
        paired_y: Tensor | None = None,
        randomness: Randomness | int = 0,
    ) -> "Vec2Vec":
        """Train on the two unpaired sets. vec2vec is unpaired, so known pairs are not used."""
        if paired_x is not None and paired_x.shape[0] > 0:
            raise ValueError("vec2vec is only used without known pairs.")
        if not isinstance(randomness, Randomness):
            randomness = Randomness(randomness)
        enable_deterministic_algorithms()
        samples_x, samples_y = samples_x.float().to(self.device), samples_y.float().to(self.device)
        batch_size = min(self.batch_size, samples_x.shape[0], samples_y.shape[0])
        with randomness.global_generators():
            self.build(samples_x.shape[1], samples_y.shape[1])
            stream_x = BatchStream(samples_x.shape[0], batch_size, self.device)
            stream_y = BatchStream(samples_y.shape[0], batch_size, self.device)
            self.translator.train()
            for _ in trange(self.num_steps, desc="vec2vec", disable=not self.verbose):
                self.training_step(samples_x[stream_x.next()], samples_y[stream_y.next()])
        self.translator.eval()
        return self

    def training_step(
        self,
        batch_x: Tensor,
        batch_y: Tensor,
    ) -> dict[str, Tensor]:
        """One update of the three discriminators and the translator on one batch of each space."""
        translator = self.translator
        batch_x, batch_y = normalize(batch_x, p=2, dim=1), normalize(batch_y, p=2, dim=1)
        adapted_x, adapted_y = translator.input_adapter_x(batch_x), translator.input_adapter_y(batch_y)
        latent_x, latent_y = translator.backbone(adapted_x), translator.backbone(adapted_y)
        reconstructed_x = translator.output_adapter_x(latent_x)
        reconstructed_y = translator.output_adapter_y(latent_y)
        reconstructed_x = normalize(reconstructed_x, p=2, dim=1)
        reconstructed_y = normalize(reconstructed_y, p=2, dim=1)
        translated_yx = normalize(translator.output_adapter_x(latent_y), p=2, dim=1)
        translated_xy = normalize(translator.output_adapter_y(latent_x), p=2, dim=1)

        losses = {}
        losses["discriminator_y"] = self.gan_y.update(batch_y, translated_xy.detach().clone())
        losses["gan_y"] = self.gan_y.generator_loss(translated_xy)
        losses["discriminator_x"] = self.gan_x.update(batch_x, translated_yx.detach().clone())
        losses["gan_x"] = self.gan_x.generator_loss(translated_yx)
        losses["discriminator_latent"] = self.gan_latent.update(latent_x.detach().clone(), latent_y.detach().clone())
        losses["gan_latent"] = self.gan_latent.generator_loss(latent_y)
        losses["reconstruction_x"] = cosine_loss(reconstructed_x, batch_x)
        losses["reconstruction_y"] = cosine_loss(reconstructed_y, batch_y)
        losses["vsp_x"] = vsp_loss(batch_x, reconstructed_x)
        losses["vsp_y"] = vsp_loss(batch_y, reconstructed_y)
        cycle_x = translator.translate_yx(translated_xy)
        cycle_y = translator.translate_xy(translated_yx)
        losses["cycle_vsp_x"] = vsp_loss(batch_x, cycle_x)
        losses["cycle_vsp_y"] = vsp_loss(batch_y, cycle_y)
        losses["cycle_x"] = cosine_loss(cycle_x, batch_x)
        losses["cycle_y"] = cosine_loss(cycle_y, batch_y)
        # Weight 1 for reconstruction, vsp and GAN, weight 10 for cycle and cycle vsp, each averaged over the two
        # spaces. The summation order fixes the rounding.
        loss = (
            0.5 * losses["reconstruction_x"]
            + 0.5 * losses["reconstruction_y"]
            + 0.5 * losses["vsp_x"]
            + 0.5 * losses["vsp_y"]
            + 5.0 * losses["cycle_vsp_x"]
            + 5.0 * losses["cycle_vsp_y"]
            + 5.0 * losses["cycle_x"]
            + 5.0 * losses["cycle_y"]
            + losses["gan_x"]
            + losses["gan_y"]
            + losses["gan_latent"]
        )
        self.optimizer.zero_grad()
        loss.backward()
        # Only the translator's gradients count, as in the official code.
        torch.nn.utils.clip_grad_norm_(translator.parameters(), MAX_GRADIENT_NORM)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        losses["translator"] = loss.detach()
        return losses

    @torch.inference_mode()
    def similarity(
        self,
        queries_x: Tensor,
        keys_y: Tensor,
    ) -> Tensor:
        """Inner products of the unit-length translated queries with the keys, which the loaders give at unit length."""
        queries_x = normalize(queries_x.float().to(self.device), p=2, dim=1)
        return self.translator.translate_xy(queries_x) @ keys_y.float().to(self.device).T
