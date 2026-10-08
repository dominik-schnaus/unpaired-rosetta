"""Tissue <-> MRI (TCGA glioma): the diagnostic pathology slide and the pre-operative brain MRI of the same patient.

The patients are the TCGA glioma cases of BraTS 2020 that also have a diagnostic slide on the GDC. Row ``i`` of both
files is patient ``i`` of ``cohort.csv`` (123 patients, sorted barcodes). The slide is embedded with Prov-GigaPath
(the mean over 256 random tissue patches at 20x), the MRI with DINO ViT-B/16 (the mean over five tumour slices, each
an RGB composite of the t1ce, t2 and flair sequences). The two sides only share the patient, and alignment stays at
chance (Table 12).

    pixi run -e modalities python -m modalities.tissue_mri.download  # checks BraTS 2020, pins the GDC slides
    pixi run -e modalities python -m modalities.tissue_mri.embed     # embeddings/TCGA-glioma-path-mri/
"""

from pathlib import Path

import pandas as pd

from modalities.common import EmbeddingFile, RawData
from unpaired_rosetta.embeddings import storage_root

DATASET = "TCGA-glioma-path-mri"
DATA_ROOT = storage_root() / "data" / "tissue_mri"
BRATS_ZIP = DATA_ROOT / "brats" / "BRATS-2020.zip"  # Kaggle "BraTS 2020 training data", downloaded by hand
BRATS_FOLDER = "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
COHORT = Path(__file__).with_name("cohort.csv")  # barcode, BraTS subject, grade, GDC slide file id and size

NUM_PATIENTS = 123
PATHOLOGY = EmbeddingFile(DATASET, "pathology", "gigapath", (NUM_PATIENTS, 1536), "float32")
MRI = EmbeddingFile(DATASET, "mri", "dinov1_vitb16-comp", (NUM_PATIENTS, 768), "float32")
PAIR = (PATHOLOGY, MRI)
FILES = [PATHOLOGY, MRI]

RAW_DATA = [
    RawData(
        "BraTS 2020 training data (Kaggle zip, MRI of 369 subjects)",
        download_bytes=4_468_587_642,
        stored_bytes=4_468_587_642,
    ),
    # One slide at a time. Each is deleted as soon as its patches are embedded.
    RawData("TCGA-GBM/LGG diagnostic slides (SVS) from the GDC", download_bytes=47_519_075_645, stored_bytes=0),
]


def cohort() -> pd.DataFrame:
    return pd.read_csv(COHORT)
