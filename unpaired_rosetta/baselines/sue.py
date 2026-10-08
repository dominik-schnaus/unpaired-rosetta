"""SUE (Yacobi et al., 2025): spectral embeddings of both spaces, aligned by CCA on the known pairs and an MMD network.

1. A SpectralNet per space learns the leading Laplacian eigenvectors of its kNN graph, on at most
   ``max_fit_samples`` unpaired samples.
2. CCA on the known pairs rotates both spectral embeddings into one space. CCA needs at least two pairs, so with
   fewer the embeddings stay unrotated.
3. A small residual network maps the X embeddings to minimize their MMD to the Y embeddings.

SpectralNet and the MMD network come unchanged from the official code (``external/SUE``). The hyperparameters follow
the official Flickr30k configuration, except for narrower SpectralNets, a smaller batch and the subsampling, which
our runs needed to fit the corpus.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from torch import Tensor
from torch.nn.functional import normalize

from unpaired_rosetta.determinism import torch_threads
from unpaired_rosetta.linalg import center_and_normalize
from unpaired_rosetta.randomness import Randomness

SUE_SOURCE = Path(__file__).resolve().parents[2] / "external" / "SUE" / "src"


def load_official_sue() -> tuple[type, Callable[..., tuple[Any, Any, torch.nn.Module]]]:
    """SpectralNet and the MMD fine-tuning of the official code, which imports its packages as top-level modules."""
    if not SUE_SOURCE.exists():
        raise ImportError(f"{SUE_SOURCE} is missing; run `git submodule update --init external/SUE`.")
    if str(SUE_SOURCE) not in sys.path:
        sys.path.append(str(SUE_SOURCE))
    from mmd import fine_tune_alignment_using_mmd_network
    from spectralnet import SpectralNet

    return SpectralNet, fine_tune_alignment_using_mmd_network


def subsample(
    samples: Tensor,
    size: int,
    randomness: Randomness,
) -> Tensor:
    if samples.shape[0] <= size:
        return samples
    return samples[randomness.subset(samples.shape[0], size)]


class SUE:
    def __init__(
        self,
        num_eigenvectors: int = 30,
        num_components: int = 8,  # CCA components
        hidden_widths: tuple[int, ...] = (1024, 1024, 512),
        epochs: int = 100,
        learning_rate: float = 1e-4,
        batch_size: int = 2048,
        num_neighbors: int = 30,  # neighbors in the affinity graph
        scale_neighbor: int = 15,  # the Gaussian scale is the median distance to this neighbor
        max_fit_samples: int = 6000,
        mmd_epochs: int = 100,
        mmd_batch_size: int = 32,
        mmd_num_scales: int = 3,
        device: str = "cpu",
    ) -> None:
        self.num_eigenvectors = num_eigenvectors
        self.num_components = num_components
        self.hidden_widths = hidden_widths
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.scale_neighbor = scale_neighbor
        self.max_fit_samples = max_fit_samples
        self.mmd_epochs = mmd_epochs
        self.mmd_batch_size = mmd_batch_size
        self.mmd_num_scales = mmd_num_scales
        self.device = torch.device(device)
        self.mean_x = self.mean_y = None
        self.spectral_net_x = self.spectral_net_y = None
        self.rotation_x = self.rotation_y = None  # CCA rotations
        self.mmd_network = None

    def build_spectral_net(
        self,
        spectral_net_class: type,
        num_samples: int,
        input_dim: int,
    ) -> Any:
        # The official constructor puts the network on cuda:1. Building it on the CPU gives the same initial weights.
        with mock.patch("torch.cuda.is_available", return_value=False):
            net = spectral_net_class(
                n_clusters=self.num_eigenvectors,
                should_use_ae=False,
                should_use_siamese=False,
                spectral_hiddens=[*self.hidden_widths, self.num_eigenvectors],
                spectral_epochs=self.epochs,
                spectral_lr=self.learning_rate,
                spectral_batch_size=min(self.batch_size, num_samples),
                spectral_n_nbg=min(self.num_neighbors, num_samples - 1),
                spectral_scale_k=self.scale_neighbor,
                spectral_is_local_scale=False,
                spectral_input_dim=input_dim,
            )
        net.device = self.device
        net.spec_net = net.spec_net.to(self.device)
        return net

    @torch_threads()
    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None = None,
        paired_y: Tensor | None = None,
        randomness: Randomness | int = 0,
    ) -> "SUE":
        if not isinstance(randomness, Randomness):
            randomness = Randomness(randomness)
        spectral_net_class, fine_tune_with_mmd = load_official_sue()
        samples_x, samples_y = samples_x.to(self.device, torch.float32), samples_y.to(self.device, torch.float32)
        self.mean_x = samples_x.mean(dim=0, keepdim=True)
        self.mean_y = samples_y.mean(dim=0, keepdim=True)
        samples_x = center_and_normalize(samples_x, self.mean_x)
        samples_y = center_and_normalize(samples_y, self.mean_y)

        fit_x = subsample(samples_x, self.max_fit_samples, randomness)
        fit_y = subsample(samples_y, self.max_fit_samples, randomness)
        with randomness.global_generators():  # the official code draws from the global generators
            self.spectral_net_x = self.build_spectral_net(spectral_net_class, *fit_x.shape)
            self.spectral_net_y = self.build_spectral_net(spectral_net_class, *fit_y.shape)
            self.spectral_net_x.fit(fit_x)
            self.spectral_net_y.fit(fit_y)
            embeddings_x = self.spectral_net_x.transform(samples_x)
            embeddings_y = self.spectral_net_y.transform(samples_y)

            if paired_x is not None and paired_x.shape[0] > 1:
                embeddings_x, embeddings_y = self.fit_cca(embeddings_x, embeddings_y, paired_x, paired_y)
            _, _, self.mmd_network = fine_tune_with_mmd(
                X=embeddings_x,
                Y=embeddings_y,
                device=self.device,
                epochs=self.mmd_epochs,
                batch_size=self.mmd_batch_size,
                n_scales=self.mmd_num_scales,
            )
        return self

    def fit_cca(
        self,
        embeddings_x: np.ndarray,
        embeddings_y: np.ndarray,
        paired_x: Tensor,
        paired_y: Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit CCA on the spectral embeddings of the known pairs and rotate all embeddings."""
        paired_x = center_and_normalize(paired_x.to(self.device, torch.float32), self.mean_x)
        paired_y = center_and_normalize(paired_y.to(self.device, torch.float32), self.mean_y)
        anchors_x, anchors_y = self.spectral_net_x.transform(paired_x), self.spectral_net_y.transform(paired_y)
        num_components = max(1, min(self.num_components, anchors_x.shape[1], anchors_x.shape[0] - 1))
        cca = CCA(n_components=num_components).fit(anchors_x, anchors_y)
        self.rotation_x = torch.as_tensor(cca.x_rotations_, dtype=torch.float32, device=self.device)
        self.rotation_y = torch.as_tensor(cca.y_rotations_, dtype=torch.float32, device=self.device)
        return embeddings_x @ cca.x_rotations_, embeddings_y @ cca.y_rotations_

    @torch.inference_mode()
    def embed(
        self,
        samples: Tensor,
        mean: Tensor,
        spectral_net: Any,
        rotation: Tensor | None,
    ) -> Tensor:
        samples = center_and_normalize(samples.to(self.device, torch.float32), mean)
        embeddings = torch.as_tensor(spectral_net.transform(samples), dtype=torch.float32, device=self.device)
        return embeddings if rotation is None else embeddings @ rotation

    @torch.inference_mode()
    def similarity(
        self,
        queries_x: Tensor,
        keys_y: Tensor,
    ) -> Tensor:
        mapped_x = self.mmd_network(self.embed(queries_x, self.mean_x, self.spectral_net_x, self.rotation_x))
        mapped_y = self.embed(keys_y, self.mean_y, self.spectral_net_y, self.rotation_y)
        return (normalize(mapped_x, dim=-1) @ normalize(mapped_y, dim=-1).T).cpu()
