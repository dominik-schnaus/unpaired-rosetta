"""A copy of AstroCLIP's SpecFormer without extra dependencies, for loading `specformer.ckpt`.

The upstream class in `astroclip/models/specformer.py` is a `lightning.LightningModule` with
blocks in `astroclip/modules.py`. Importing it would pull in lightning, hydra and the AstroCLIP
package for a plain transformer encoder. The code below is copied from
https://github.com/PolymathicAI/AstroCLIP (MIT) with three changes:

  * `L.LightningModule` becomes `nn.Module`, and `save_hyperparameters()` becomes a small
    namespace, so `self.hparams.max_len` and the like still work. Attribute names are kept, so
    the checkpoint's `state_dict` keys (`data_embed.*`, `blocks.N.*`, `final_layernorm.*`,
    `head.*`) load unchanged.
  * The training and validation steps and the masking helper are dropped.
  * `_init_by_depth` is kept because the constructor calls it before the weights are loaded.

`load_specformer` follows the upstream recipe:

    checkpoint = torch.load(path)
    specformer = SpecFormer(**checkpoint["hyper_parameters"])
    specformer.load_state_dict(checkpoint["state_dict"])

The AstroCLIP authors pool with `np.mean(specformer(x)["embedding"], axis=1)` (see
`downstream_tasks/property_estimation/embed_provabgs.py`). It is the mean of the final hidden
state over sequence positions.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from pathlib import Path


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias. PyTorch's own cannot turn the bias off."""

    normalized_shape: tuple[int, ...]
    eps: float

    def __init__(
        self,
        shape: int | tuple[int, ...] | torch.Size,
        eps: float = 1e-5,
        bias: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        if isinstance(shape, numbers.Integral):
            self.normalized_shape = (shape,)
        else:
            self.normalized_shape = tuple(shape)
        self.weight = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.empty(shape)) if bias else None
        self.reset_parameters()

    def reset_parameters(
        self,
    ) -> None:
        torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


class MLP(nn.Module):
    """A two-layer MLP with an expansion in the middle."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        activation: Callable | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.activation = activation if activation is not None else nn.GELU()
        self.dropout = dropout
        self.bias = bias
        self.encoder = nn.Linear(in_features, hidden_features, bias=bias)
        self.decoder = nn.Linear(hidden_features, in_features, bias=bias)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        return x


class SelfAttention(nn.Module):
    """Collection of self-attention heads."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        causal: bool,
        dropout: float,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim should be divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        self.attention = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, channels = x.shape
        if channels != self.embedding_dim:
            raise ValueError(f"Expected (..., {self.embedding_dim}), got {tuple(x.shape)}")
        query, key, value = self.attention(x).split(self.embedding_dim, dim=2)
        num_heads = self.num_heads
        head_size = channels // num_heads
        key = key.view(batch, length, num_heads, head_size).transpose(1, 2)
        query = query.view(batch, length, num_heads, head_size).transpose(1, 2)
        value = value.view(batch, length, num_heads, head_size).transpose(1, 2)
        dropout_p = self.dropout if self.training else 0
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=self.causal
        )
        out = out.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.residual_dropout(self.projection(out))


class TransformerBlock(nn.Module):
    """Layer norm, self-attention, layer norm and a two-layer MLP, each with a residual."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        causal: bool,
        dropout: float,
        bias: bool = True,
        mlp_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.layernorm1 = LayerNorm(embedding_dim, bias=bias)
        self.attention = SelfAttention(embedding_dim, num_heads, bias=bias, dropout=dropout, causal=causal)
        self.layernorm2 = LayerNorm(embedding_dim, bias=bias)
        hidden_dim = mlp_expansion * embedding_dim
        self.mlp = MLP(embedding_dim, hidden_dim, nn.GELU(), dropout=dropout, bias=bias)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x = x + self.attention(self.layernorm1(x))
        x = x + self.mlp(self.layernorm2(x))
        return x


def _init_by_depth(
    module: nn.Module,
    depth: int,
) -> None:
    """Depth-scaled truncated-normal init. The checkpoint overwrites it."""
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(-1)
        std = 1 / math.sqrt(2 * fan_in * depth)
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _HParams(dict):
    """Stand-in for lightning's `self.hparams`, a dict with attribute access."""

    def __getattr__(
        self,
        key: str,
    ) -> object:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


class SpecFormer(nn.Module):
    """AstroCLIP's spectrum encoder, a GPT-2-style transformer over sliced spectra."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_len: int,
        mask_num_chunks: int = 6,
        mask_chunk_width: int = 50,
        slice_section_length: int = 20,
        slice_overlap: int = 10,
        dropout: float = 0.1,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.hparams = _HParams(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_len=max_len,
            mask_num_chunks=mask_num_chunks,
            mask_chunk_width=mask_chunk_width,
            slice_section_length=slice_section_length,
            slice_overlap=slice_overlap,
            dropout=dropout,
            norm_first=norm_first,
        )

        self.data_embed = nn.Linear(input_dim, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    causal=False,
                    dropout=dropout,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = LayerNorm(embed_dim, bias=True)
        self.head = nn.Linear(embed_dim, input_dim, bias=True)

        self._reset_parameters_datapt()

    def forward(
        self,
        x: Tensor,
    ) -> dict[str, Tensor]:
        x = self.preprocess(x)
        return self.forward_without_preprocessing(x)

    def forward_without_preprocessing(
        self,
        x: Tensor,
    ) -> dict[str, Tensor]:
        length = x.shape[1]
        if length > self.hparams.max_len:
            raise ValueError(f"Cannot forward sequence of length {length}, block size is only {self.hparams.max_len}")
        positions = torch.arange(0, length, dtype=torch.long, device=x.device)
        x = self.dropout(self.data_embed(x) + self.position_embed(positions))
        for block in self.blocks:
            x = block(x)
        x = self.final_layernorm(x)
        return {"reconstructions": self.head(x), "embedding": x}

    def preprocess(
        self,
        x: Tensor,
    ) -> Tensor:
        """Standardize each spectrum and slice it into overlapping sections.

        A padded first row carries the removed mean and std, so the encoder keeps the flux scale.
        """
        std, mean = x.std(1, keepdim=True).clip_(0.2), x.mean(1, keepdim=True)
        x = (x - mean) / std
        x = self._slice(x)
        x = F.pad(x, pad=(2, 0, 1, 0), mode="constant", value=0)
        x[:, 0, 0] = (mean.squeeze() - 2) / 2
        x[:, 0, 1] = (std.squeeze() - 2) / 8
        return x

    def _reset_parameters_datapt(
        self,
    ) -> None:
        for embedding in [self.data_embed, self.position_embed]:
            std = 1 / math.sqrt(self.hparams.embed_dim)
            nn.init.trunc_normal_(embedding.weight, std=std, a=-3 * std, b=3 * std)
        self.blocks.apply(lambda m: _init_by_depth(m, self.hparams.num_layers))
        self.head.apply(lambda m: _init_by_depth(m, 1 / 2))

    def _slice(
        self,
        x: Tensor,
    ) -> Tensor:
        start_indices = np.arange(
            0,
            x.shape[1] - self.hparams.slice_overlap,
            self.hparams.slice_section_length - self.hparams.slice_overlap,
        )
        sections = [x[:, start : start + self.hparams.slice_section_length].transpose(1, 2) for start in start_indices]
        if sections[-1].shape[1] < self.hparams.slice_section_length:
            sections.pop(-1)
        return torch.cat(sections, 1)


def _stub_lightning() -> None:
    """Let `specformer.ckpt` load without lightning installed.

    The checkpoint's `hyper_parameters` entry is a `lightning.fabric.utilities.data.AttributeDict`,
    a plain dict subclass. It is the only lightning symbol in the pickle stream (checked by
    walking the stream), so a stub module is enough. It keeps a ~200 MB training framework out of
    the environment. The stub is only added when lightning is missing.
    """
    import sys
    import types

    if "lightning" in sys.modules:
        return
    try:
        import lightning  # noqa: F401

        return
    except ImportError:
        pass

    class AttributeDict(dict):
        """Dict with attribute access, like lightning's own."""

        def __getattr__(
            self,
            key: str,
        ) -> object:
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

    path = "lightning.fabric.utilities.data"
    AttributeDict.__module__ = path
    parts = path.split(".")
    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        module = types.ModuleType(name)
        if depth > 1:
            setattr(sys.modules[".".join(parts[: depth - 1])], parts[depth - 1], module)
        sys.modules[name] = module
    sys.modules[path].AttributeDict = AttributeDict


def load_specformer(
    checkpoint_path: str | Path,
) -> SpecFormer:
    """Rebuild SpecFormer from a lightning checkpoint, as upstream does."""
    _stub_lightning()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = SpecFormer(**dict(checkpoint["hyper_parameters"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model
