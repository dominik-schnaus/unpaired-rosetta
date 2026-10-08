"""The class-matching benchmark of Schnaus et al. (2025, "It's a (Blind) Match!"), used for Figure 7.

The task matches the 100 classes of CIFAR-100 between vision and language without any pairs. The vision side
averages the CLIP ViT-L/14@336 embeddings of a random half of the test images of every class. The language side
averages the all-mpnet-base-v2 embeddings of 18 CLIP prompts per class name. For each problem size ``C`` in 10, 20,
..., 100, a p-dispersion-sum problem picks the ``C`` classes whose vision and language kernels are most alike
(Gurobi, one hour). The QAP then asks for the permutation of these classes that minimizes the Gromov-Wasserstein
cost between the two standardized distance kernels.

Files under ``<root>/qap_benchmark``: the vision embeddings ``[10000, 768]``, the language embeddings
``[100, 18, 768]``, the test labels ``[10000]`` (float64 and int64, as itsamatch stored them), and one subset per
size. ``materialize`` copies them from the itsamatch storage. They were computed there with
``itsamatch/extract_embeddings.py`` (OpenAI ``clip`` package, ``sentence-transformers``) and
``itsamatch/experiments/larger_scale_matching.py`` (``compute_missing_subsets``). The ``compute_*`` functions
recompute them.
"""

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.functional import normalize

from unpaired_rosetta.embeddings import storage_root
from unpaired_rosetta.settings import itsamatch_root

SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SEED = 734796314  # itsamatch's ``get_seeds(42, 1)[0]``, the same as ``SEEDS[0]`` of this repository
SUBSET_TIME_LIMIT = 3600  # seconds of Gurobi per subset
VISION_MODEL = "clip_vit-l14@336"
LANGUAGE_MODEL = "sentencet_all-mpnet-base-v2"
PROMPT_HASH = "b953684b7ed3a62cbea9fe4e30935ee2"  # itsamatch's hash of the 18 CIFAR prompts
# Embeddings and class subsets of itsamatch (github.com/dominik-schnaus/itsamatch), which the benchmark reuses.
ITSAMATCH_EMBEDDINGS = itsamatch_root() / "embeddings" / "CIFAR-100"
ITSAMATCH_SUBSETS = itsamatch_root() / "subsets" / "GromovWasserstein_" / f"CIFAR-100_{PROMPT_HASH}"

PROMPTS = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
]


def folder() -> Path:
    return storage_root() / "qap_benchmark"


def vision_path() -> Path:
    return folder() / f"{VISION_MODEL}.pt"


def language_path() -> Path:
    return folder() / f"{LANGUAGE_MODEL}.pt"


def labels_path() -> Path:
    return folder() / "labels.pt"


def subset_path(
    size: int,
) -> Path:
    return folder() / "subsets" / f"{size}.pt"


def materialize() -> None:
    """Copy the paper's embeddings, labels and subsets. Each size uses the first of its ten pooled subsets."""
    copies = {
        vision_path(): ITSAMATCH_EMBEDDINGS / "vision" / f"{VISION_MODEL}.pt",
        language_path(): ITSAMATCH_EMBEDDINGS / f"language_{PROMPT_HASH}" / f"{LANGUAGE_MODEL}.pt",
        labels_path(): ITSAMATCH_EMBEDDINGS / "labels.pt",
    }
    for size in SIZES:
        subset_file = f"{VISION_MODEL}_{LANGUAGE_MODEL}_{size}_0_{SUBSET_TIME_LIMIT}.pt"
        copies[subset_path(size)] = ITSAMATCH_SUBSETS / subset_file
    for target, source in copies.items():
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.parent.name == "subsets":
            torch.save(torch.load(source, weights_only=False)["subset"], target)
        else:
            shutil.copyfile(source, target)


@dataclass
class ClassMatchingProblem:
    """``min_pi sum_{i,k} cost1[i, k] cost2[pi(i), pi(k)] + constant``, the Gromov-Wasserstein cost.

    It is the cost of matching vision class ``i`` to language class ``pi(i)``. The identity is the ground truth.
    """

    cost1: Tensor
    cost2: Tensor
    constant: Tensor
    vision_kernel: Tensor  # standardized kernels, for solvers that need the full cost (Gurobi)
    language_kernel: Tensor


def standardize_kernel(
    kernel: Tensor,
) -> Tensor:
    """Standardize the off-diagonal entries (mean 0, standard deviation 1) and set the diagonal to zero."""
    off_diagonal = torch.ones_like(kernel, dtype=torch.bool).fill_diagonal_(False)
    return ((kernel - kernel[off_diagonal].mean()) / kernel[off_diagonal].std()).fill_diagonal_(0)


def distance_kernel(
    embeddings: Tensor,
) -> Tensor:
    """Pairwise Euclidean distances, the kernel of the Gromov-Wasserstein distortion."""
    return torch.cdist(embeddings, embeddings, p=2)


def cka_kernel(
    embeddings: Tensor,
) -> Tensor:
    """LocalCKA kernel as itsamatch implements it.

    Rows are scaled to unit standard deviation, and the linear kernel is row-centered and scaled so that
    ``sum K * K^T = 1`` (elementwise product).
    """
    embeddings = embeddings / embeddings.std(dim=1, keepdim=True)
    inner_product = embeddings @ embeddings.T
    centered = inner_product - inner_product.mean(dim=1, keepdim=True)
    return centered / (centered * centered.T).sum().sqrt()


def load_embeddings() -> tuple[Tensor, Tensor, Tensor]:
    """Vision embeddings, one normalized language embedding per class (mean over the prompts), and the labels."""
    vision = torch.load(vision_path())
    language = normalize(torch.load(language_path()).nanmean(dim=1), dim=-1)
    return vision, language, torch.load(labels_path())


def class_means(
    vision: Tensor,
    labels: Tensor,
    seed: int = SEED,
) -> Tensor:
    """Mean vision embedding of every class over a random half of the images (drawn with its own generator)."""
    half = torch.randperm(vision.shape[0], generator=torch.Generator().manual_seed(seed))[: vision.shape[0] // 2]
    vision, labels_half = vision[half], labels[half]
    return torch.stack([vision[labels_half == label].mean(dim=0) for label in labels.unique(sorted=True)])


def kernels(
    size: int,
    kernel: Callable[[Tensor], Tensor] = distance_kernel,
) -> tuple[Tensor, Tensor]:
    """Standardized vision and language kernels of the ``size`` classes of the subset.

    As in itsamatch, the language kernel is standardized over all 100 classes before the subset is taken. The vision
    kernel is standardized over the subset only.
    """
    vision, language, labels = load_embeddings()
    subset = torch.as_tensor(torch.load(subset_path(size)))
    language_kernel = standardize_kernel(kernel(language))[subset][:, subset]
    vision_kernel = standardize_kernel(kernel(normalize(class_means(vision, labels)[subset], dim=-1)))
    return vision_kernel, language_kernel


def gromov_wasserstein_problem(
    size: int,
) -> ClassMatchingProblem:
    """Gromov-Wasserstein problem of the ``size`` classes.

    The squared loss ``(a - b)^2 = a^2 + b^2 - a*2b`` splits into ``cost1 = -K_V``, ``cost2 = 2 K_L`` and the
    constant ``sum K_V^2 + sum K_L^2``, as in itsamatch's ``SquaredLoss``.
    """
    vision_kernel, language_kernel = kernels(size)
    constant = (vision_kernel**2).sum() + (language_kernel**2).sum()
    return ClassMatchingProblem(-vision_kernel, 2 * language_kernel, constant, vision_kernel, language_kernel)


def cka_problem(
    size: int,
) -> ClassMatchingProblem:
    """LocalCKA's problem: maximize ``sum K_V[i, k] K_L[pi(i), pi(k)]`` for the standardized CKA kernels."""
    vision_kernel, language_kernel = kernels(size, cka_kernel)
    constant = torch.zeros((), dtype=vision_kernel.dtype)
    return ClassMatchingProblem(-vision_kernel, language_kernel, constant, vision_kernel, language_kernel)


def gromov_wasserstein_cost(
    size: int,
    permutation: Tensor,
) -> float:
    """``sum_{i,k} (K_V[i, k] - K_L[pi(i), pi(k)])^2``, the cost Figure 7 reports for the permutation of any solver."""
    vision_kernel, language_kernel = kernels(size)
    return ((vision_kernel - language_kernel[permutation][:, permutation]) ** 2).sum().item()


def compute_subset(
    size: int,
    time_limit: float = SUBSET_TIME_LIMIT,
) -> list[bool]:
    """The ``size`` classes with the smallest Gromov-Wasserstein cost under the identity matching.

    The problem is ``min_{b in {0,1}^100, sum b = size} sum_{i,j} b_i b_j (c + c^T)_{ij}`` with
    ``c = (K_V - K_L)^2`` over all images. It is a p-dispersion-sum problem, solved with Gurobi as in
    ``larger_scale_matching.get_subset`` of itsamatch.
    """
    import gurobipy as gp
    from gurobipy import GRB

    vision, language, labels = load_embeddings()
    vision_means = torch.stack([vision[labels == label].mean(dim=0) for label in labels.unique(sorted=True)])
    vision_kernel = standardize_kernel(distance_kernel(normalize(vision_means, dim=-1)))
    language_kernel = standardize_kernel(distance_kernel(language))
    cost = ((vision_kernel - language_kernel) ** 2).numpy()
    if size == cost.shape[0]:
        return [True] * size
    cost = cost + cost.T
    model = gp.Model("subset selection")
    model.Params.OutputFlag = 0
    model.Params.PoolSearchMode = 2  # a pool of the 10 best subsets, the paper uses the best
    model.Params.TuneCriterion = 2
    model.Params.CliqueCuts = 2
    model.Params.Heuristics = 0.5
    model.Params.TimeLimit = time_limit
    chosen = model.addMVar(shape=(len(cost),), lb=0.0, ub=1.0, vtype=GRB.BINARY)
    model.setObjective(
        gp.quicksum(cost[i, j] * chosen[i] * chosen[j] for i in range(len(cost)) for j in range(len(cost))),
        GRB.MINIMIZE,
    )
    model.addConstr(chosen.sum() == size)
    model.optimize()
    model.Params.SolutionNumber = 0
    indices = chosen.Xn.argsort()[-size:]
    model.dispose()
    return [index in indices for index in range(len(cost))]


def compute_language_embeddings(
    class_names: list[str],
) -> Tensor:
    """``[classes, prompts, 768]`` float64 embeddings of the 18 prompts of every class (sentence-transformers)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-mpnet-base-v2")
    texts = [prompt.format(name) for name in class_names for prompt in PROMPTS]
    embeddings = model.encode(texts, convert_to_tensor=True).cpu().double()
    return embeddings.view(len(class_names), len(PROMPTS), -1)


def compute_vision_embeddings(
    data_root: Path,
    batch_size: int = 64,
) -> tuple[Tensor, Tensor]:
    """``[10000, 768]`` float64 CLIP ViT-L/14@336 image embeddings of the CIFAR-100 test set and its labels.

    Needs OpenAI's ``clip`` package (``pip install git+https://github.com/openai/CLIP``), which is not part of the
    default environment.
    """
    import clip
    from torchvision.datasets import CIFAR100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-L/14@336px", device=device)
    dataset = CIFAR100(root=data_root, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)
    embeddings, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            embeddings.append(model.encode_image(images.to(device)).cpu().double())
            labels.append(targets)
    return torch.cat(embeddings), torch.cat(labels)
