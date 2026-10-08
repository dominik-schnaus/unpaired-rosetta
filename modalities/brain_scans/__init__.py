"""Neuroscience: cross-subject fMRI alignment on the Natural Scenes Dataset (NSD), in the setting of
Marcos-Manchon et al. (2026, "Platonic Brain").

Their pipeline (``pipeline/run_pipeline.sh`` runs the authors' code) fits one encoder per subject. It maps the
subject's fMRI responses (nsdgeneral voxels) to 128 dimensions with a linear encoder refined by a residual MLP. The
outputs are standardized and averaged over the repetitions of an image. Each of the eight subjects saw about 9000
private images (the unpaired training set, different images for every subject) and the shared images, of which 907
were seen by all eight subjects (subjects 3, 4, 6 and 8 did not complete all sessions). These 907 images are the
paired validation set::

    embeddings/NSD-private/sub01/mlp128.pt      [9000, 128]  responses of subject 1 to its private images
    embeddings/NSD-shared907/sub01/mlp128.pt    [907, 128]   responses to the shared images (sorted NSD ids, so row i
                                                             is the same image for every subject)

``materialize`` writes these files from the pipeline's stage-3 outputs.
"""

import os
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from modalities.common import EmbeddingFile
from unpaired_rosetta.embeddings import embedding_path, storage_root

ENCODER = "mlp128"
TRAINING_DATASET = "NSD-private"
VALIDATION_DATASET = "NSD-shared907"
SUBJECTS = [f"sub{index:02d}" for index in range(1, 9)]
SUBJECT_PAIRS = list(permutations(SUBJECTS, 2))  # the 56 ordered pairs
NUM_PRIVATE_IMAGES = {
    "sub01": 9000,
    "sub02": 9000,
    "sub03": 8481,
    "sub04": 8302,
    "sub05": 9000,
    "sub06": 8481,
    "sub07": 9000,
    "sub08": 8302,
}
NUM_SHARED_IMAGES = 907


def pipeline_root() -> Path:
    """Data and outputs of the pipeline.

    The folder is ``$NSD_DATASET`` (the variable the authors' code reads), by default ``<storage root>/data/nsd``.
    """
    return Path(os.environ.get("NSD_DATASET", storage_root() / "data" / "nsd"))


def stage_three_payload(
    subject: str,
) -> dict[str, object]:
    """Standardized, repetition-averaged embeddings of one subject.

    ``Z_train``/``labels_train`` hold the private images and ``Z_test``/``labels_test`` the shared ones (labels are
    1-based NSD image ids).
    """
    folder = pipeline_root() / "runs" / "1mm" / "mlp_embeddings_standarized"
    path = folder / f"avg_ws_mlp_v1_128_768_sub-{subject[3:]}.pt"
    return torch.load(path, weights_only=False, map_location="cpu")


def embedding_files() -> list[EmbeddingFile]:
    training = [
        EmbeddingFile(TRAINING_DATASET, subject, ENCODER, (NUM_PRIVATE_IMAGES[subject], 128), "float32")
        for subject in SUBJECTS
    ]
    validation = [
        EmbeddingFile(VALIDATION_DATASET, subject, ENCODER, (NUM_SHARED_IMAGES, 128), "float32") for subject in SUBJECTS
    ]
    return training + validation


def materialize() -> None:
    """Write the training and validation embeddings of every subject from the stage-3 payloads."""
    payloads = {subject: stage_three_payload(subject) for subject in SUBJECTS}
    shared = np.asarray(payloads[SUBJECTS[0]]["labels_test"])
    for payload in payloads.values():
        shared = np.intersect1d(shared, np.asarray(payload["labels_test"]))  # sorted ids seen by every subject
    for subject, payload in payloads.items():
        row_of = {int(image): row for row, image in enumerate(np.asarray(payload["labels_test"]))}
        validation = torch.as_tensor(payload["Z_test"]).float()[[row_of[int(image)] for image in shared]]
        training = torch.as_tensor(payload["Z_train"]).float()
        for dataset, embeddings in [(TRAINING_DATASET, training), (VALIDATION_DATASET, validation)]:
            path = embedding_path(dataset, subject, ENCODER)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embeddings, path)
            print(f"wrote {path} {tuple(embeddings.shape)}")
