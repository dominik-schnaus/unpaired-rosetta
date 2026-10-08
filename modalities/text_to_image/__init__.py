"""Text-to-image generation through the alignment (Sec. 4.5, App. B and C.2).

A sentence encoder (MPNet or Contriever) embeds a caption. An aligner fitted on MS COCO maps it into the image
embedding space. A representation-autoencoder (RAE) diffusion model, conditioned only on image embeddings, generates
the image. The diffusion model is trained on ImageNet images and never sees a caption, so language enters only
through the aligner.

The code runs in three pixi environments because their pinned packages cannot be solved together (see
``pyproject.toml``):

* ``text2image``: RAE training, caption embeddings and sampling (``train_rae.sbatch``, ``sample.py``).
* ``scale-rae``: the paired reference model (``scale_rae.py``).
* ``t2i-metrics``: the four scores (``scores.py``, ``tifa.py``).

This module only holds names and paths, so every environment can import it.
"""

import hashlib
from pathlib import Path

from modalities.image_captions import language_file, vision_file
from unpaired_rosetta.embeddings import storage_root

PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parents[1]
RAE_ROOT = REPOSITORY / "external" / "RAE"  # Submodule, patched by setup.sh.
SCALE_RAE_ROOT = REPOSITORY / "external" / "Scale-RAE"
RAE_CONFIG = PACKAGE / "rae" / "DiTDH-S_DINOv2-B_raemean.yaml"

# Aligner spaces: the mean-pooled RAE encoder latent (DINOv2 ViT-B/14 with registers), which conditions the diffusion
# model, and two sentence encoders.
TRAINING_DATASET = "coco_train2014"
VISION_MODEL = "rae_dinov2_base_mean"
LANGUAGE_MODELS = {"mpnet": "sentence-transformers/all-mpnet-base-v2", "contriever": "facebook/contriever"}
LANGUAGE_NAMES = {"mpnet": "MPNet", "contriever": "Contriever"}
EMBEDDING_FILES = [vision_file(TRAINING_DATASET, VISION_MODEL)] + [
    language_file(TRAINING_DATASET, model) for model in LANGUAGE_MODELS
]

# The eight captions of the grids (Figs. 6d and 15-18).
CAPTIONS = [
    "A dog is running through a field.",
    "A group of people are sitting at a table.",
    "A beautiful sunset over the mountains.",
    "A person is riding a bicycle on a city street.",
    "A cat is playing with a ball of yarn.",
    "A train travelling down the tracks near a station.",
    "A plate of food with vegetables and meat.",
    "A bird perched on a tree branch.",
]

# Sampling (App. B): 250 Euler steps of the probability-flow ODE with guidance scale 1.0, so the unconditional branch
# is never run.
NUM_STEPS = 250
LATENT_SIZE = (768, 16, 16)
PROMPT_NOISE_SEED = 734796314


def prompt_seed(
    index: int,
) -> int:
    """Noise seed of benchmark prompt ``index``. It is the same for each row of Tables 6 and 7 and each batch."""
    return int.from_bytes(hashlib.sha256(f"{PROMPT_NOISE_SEED}:{index}".encode()).digest()[:8], "little") % 2**63


def rae_weights() -> Path:
    """Pretrained RAE decoder and latent statistics (``nyu-visionx/RAE-collections``). ``setup.sh`` links them as
    ``external/RAE/models``."""
    return storage_root() / "data" / "rae"


def diffusion_checkpoint() -> Path:
    """The paper's diffusion transformer. ``train_rae.sbatch`` wrote it at step 2,010,000, in the last of 400 epochs.
    Its EMA weights are used."""
    return storage_root() / "checkpoints" / "rae" / "ditdh-s_dinov2b_raemean" / "checkpoints" / "ep-0000399.pt"
