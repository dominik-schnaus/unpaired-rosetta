"""Compare our SUE aligner with the official training pipeline (``Trainer.fit`` and ``Trainer._get_test_embeddings``
of github.com/shaham-lab/SUE, cloned by ``tests/sue/clone.sh``) on small synthetic data on the CPU.

The official ``Trainer`` fits both SpectralNets on the whole training set, whose last ``n_parallel`` rows are the
known pairs. We give our aligner the same rows as unpaired samples and the same last rows as known pairs. Each
SpectralNet sees all rows (no subsampling), so both run the same computation on the same random streams.

One difference is on purpose. The ``Trainer`` takes the spectral embeddings of the known pairs from those of all
training rows. Our known pairs are not among the unpaired samples and are embedded on their own. The matrix products
then differ in the last bits, and the MMD training makes this larger. The tests therefore fit the official CCA on the
separately embedded pairs. ``test_separately_embedded_pairs_match_the_sliced_ones`` bounds that difference.
"""

import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from unpaired_rosetta.baselines.sue import SUE, SUE_SOURCE, load_official_sue
from unpaired_rosetta.determinism import FIT_THREADS
from unpaired_rosetta.linalg import center_and_normalize
from unpaired_rosetta.randomness import Randomness

OFFICIAL = Path(__file__).parent / "official"
pytestmark = [
    pytest.mark.official,
    pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/sue/clone.sh"),
    pytest.mark.skipif(not SUE_SOURCE.exists(), reason="run git submodule update --init external/SUE"),
]

SEED = 3
NUM_PAIRS = 20
HYPERPARAMETERS = dict(
    num_eigenvectors=6,
    num_components=4,
    hidden_widths=(32, 32),
    epochs=5,
    learning_rate=1e-3,
    batch_size=64,
    num_neighbors=10,
    scale_neighbor=5,
    mmd_epochs=100,
    mmd_batch_size=32,
    mmd_num_scales=3,  # the MMD settings of the Trainer
)


def official_config(
    input_dim: int,
) -> dict[str, object]:
    h = HYPERPARAMETERS
    return {
        "n_clusters": h["num_eigenvectors"],
        "should_use_ae": False,
        "should_use_siamese": False,
        "spectral_batch_size": h["batch_size"],
        "spectral_is_local_scale": False,
        "spectral_n_nbg": h["num_neighbors"],
        "spectral_scale_k": h["scale_neighbor"],
        "spectral_epochs": h["epochs"],
        "spectral_lr": h["learning_rate"],
        "spectral_hiddens": [*h["hidden_widths"], h["num_eigenvectors"]],
        "spectral_input_dim": input_dim,
    }


def import_official_trainer() -> types.ModuleType:
    """Import ``trainer`` of the clone with its own ``spectralnet`` and ``mmd`` modules.

    The modules of our submodule are put back afterwards. ``fit`` never uses the encoders, datasets and plotting that
    ``trainer`` imports, so they are stubs.
    """
    load_official_sue()  # the modules of our submodule, restored below
    names = [name for name in sys.modules if name.split(".")[0] in ("spectralnet", "mmd", "trainer")]
    saved = {name: sys.modules.pop(name) for name in names}
    stubs = {"umap": dict(UMAP=None), "general_utils": dict(calc_recall=None), "data": {}, "encoders": {}}
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        saved.setdefault(name, sys.modules.get(name))
        sys.modules[name] = module
    sys.path.insert(0, str(OFFICIAL / "src"))
    try:
        import trainer

        return trainer
    finally:
        sys.path.remove(str(OFFICIAL / "src"))
        for name in [name for name in sys.modules if name.split(".")[0] in ("spectralnet", "mmd", "trainer", *stubs)]:
            del sys.modules[name]
        sys.modules.update({name: module for name, module in saved.items() if module is not None})


@pytest.fixture(scope="module", autouse=True)
def fit_threads() -> Iterator[None]:
    """Run the official code on ``FIT_THREADS`` PyTorch threads like our fit, so that the sums are identical."""
    num_threads = torch.get_num_threads()
    torch.set_num_threads(FIT_THREADS)
    yield
    torch.set_num_threads(num_threads)


@pytest.fixture(scope="module")
def data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    latent = torch.randn(10, 4, generator=generator)[torch.randint(0, 10, (400,), generator=generator)]
    latent = latent + 0.3 * torch.randn(400, 4, generator=generator)
    x = latent @ torch.randn(4, 12, generator=generator) + 0.1 * torch.randn(400, 12, generator=generator)
    y = latent @ torch.randn(4, 16, generator=generator) + 0.1 * torch.randn(400, 16, generator=generator)
    return x[:300], y[:300], x[300:], y[300:]


@pytest.fixture(scope="module")
def fitted(
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[SUE, Any, tuple[np.ndarray, np.ndarray]]:
    """Our aligner and the official trainer, fitted on the same data from the same seed."""
    x, y, test_x, test_y = data
    ours = SUE(**HYPERPARAMETERS).fit(x, y, x[-NUM_PAIRS:], y[-NUM_PAIRS:], Randomness(SEED))

    trainer = import_official_trainer()
    trainer.MMDNet = lambda *args: torch.nn.Identity()  # fit replaces this MMD net, so it is never used
    mean_x, mean_y = x.mean(dim=0, keepdim=True), y.mean(dim=0, keepdim=True)
    pairs_x, pairs_y = center_and_normalize(x[-NUM_PAIRS:], mean_x), center_and_normalize(y[-NUM_PAIRS:], mean_y)

    class CCAOfSeparatelyEmbeddedPairs(trainer.CCA):
        def fit(
            self,
            X: np.ndarray,
            Y: np.ndarray,
        ) -> object:
            return super().fit(official.spectralnet1.transform(pairs_x), official.spectralnet2.transform(pairs_y))

    trainer.CCA = CCAOfSeparatelyEmbeddedPairs
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)  # the official SpectralNet asks for cuda:1
        official = trainer.Trainer(
            n_parallel=NUM_PAIRS,
            n_components=HYPERPARAMETERS["num_components"],
            device=torch.device("cpu"),
            configs={"spectralnets": [official_config(x.shape[1]), official_config(y.shape[1])]},
        )
        official.fit([center_and_normalize(x, mean_x), center_and_normalize(y, mean_y)])
        official_test = official._get_test_embeddings(
            [center_and_normalize(test_x, mean_x), center_and_normalize(test_y, mean_y)]
        )
    return ours, official, official_test


def test_submodule_is_the_unmodified_official_code() -> None:
    for path in sorted((OFFICIAL / "src").rglob("*.py")):
        assert (SUE_SOURCE / path.relative_to(OFFICIAL / "src")).read_bytes() == path.read_bytes(), path


def test_spectral_nets_equal_official(
    fitted: tuple[SUE, Any, tuple[np.ndarray, np.ndarray]],
) -> None:
    ours, official, _ = fitted
    pairs = [(ours.spectral_net_x, official.spectralnet1), (ours.spectral_net_y, official.spectralnet2)]
    for net, official_net in pairs:
        for name, value in net.spec_net.state_dict().items():
            assert torch.equal(value, official_net.spec_net.state_dict()[name]), name
        assert torch.equal(net.spec_net.orthonorm_weights, official_net.spec_net.orthonorm_weights)
        np.testing.assert_array_equal(net.Q, official_net.Q)


def test_separately_embedded_pairs_match_the_sliced_ones(
    fitted: tuple[SUE, Any, tuple[np.ndarray, np.ndarray]],
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    _, official, _ = fitted
    x, _, _, _ = data
    x = center_and_normalize(x, x.mean(dim=0, keepdim=True))
    sliced = official.spectralnet1.transform(x)[-NUM_PAIRS:]
    np.testing.assert_allclose(official.spectralnet1.transform(x[-NUM_PAIRS:]), sliced, rtol=1e-5, atol=1e-6)


def test_cca_rotations_equal_official(
    fitted: tuple[SUE, Any, tuple[np.ndarray, np.ndarray]],
) -> None:
    ours, official, _ = fitted
    assert torch.equal(ours.rotation_x, torch.as_tensor(official.projection1, dtype=torch.float32))
    assert torch.equal(ours.rotation_y, torch.as_tensor(official.projection2, dtype=torch.float32))


def test_mmd_network_equals_official(
    fitted: tuple[SUE, Any, tuple[np.ndarray, np.ndarray]],
) -> None:
    ours, official, _ = fitted
    for name, value in ours.mmd_network.state_dict().items():
        assert torch.equal(value, official.mmd_model.state_dict()[name]), name


def test_similarity_equals_official(
    fitted: tuple[SUE, Any, tuple[np.ndarray, np.ndarray]],
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    # The Trainer rotates in float64 NumPy. We rotate in float32, like the runs of the paper.
    ours, _, (official_x, official_y) = fitted
    _, _, test_x, test_y = data
    official_x = official_x / np.linalg.norm(official_x, axis=1, keepdims=True)
    official_y = official_y / np.linalg.norm(official_y, axis=1, keepdims=True)
    np.testing.assert_allclose(ours.similarity(test_x, test_y).double().numpy(), official_x @ official_y.T, atol=1e-5)


def test_fewer_than_two_pairs_skip_cca(
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x, y, test_x, test_y = data
    aligner = SUE(**HYPERPARAMETERS | dict(epochs=1, mmd_epochs=1)).fit(x, y, x[-1:], y[-1:], Randomness(SEED))
    assert aligner.rotation_x is None
    assert aligner.similarity(test_x, test_y).shape == (100, 100)
