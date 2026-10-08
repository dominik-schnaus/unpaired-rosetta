"""Tissue <-> CT (CPTAC): a diagnostic pathology slide and a CT scan of the same cancer patient.

Row ``i`` of both files is patient ``i`` of ``patients.txt`` (sorted patient ids). The slide is embedded with
H-Optimus-0 (the mean over up to 196 tissue tiles at about 20x), the CT volume with CT-FM. The two sides only share
the patient, so the pair tests whether a patient-level match can be found at all. Table 12 shows it cannot.

    pixi run -e modalities python -m modalities.tissue_ct.download   # DICOM from the IDC -> tiles and volumes
    pixi run -e modalities python -m modalities.tissue_ct.embed      # embeddings/CPTAC-path-ct/{pathology,ct}/
"""

from pathlib import Path

from modalities.common import EmbeddingFile, RawData
from unpaired_rosetta.embeddings import storage_root

DATASET = "CPTAC-path-ct"
DATA_ROOT = storage_root() / "data" / "tissue_ct"
CACHE = DATA_ROOT / "cache"  # <patient>.pt: {"tiles": [T, 3, 224, 224] uint8, "ct": [96, 224, 224] float16, ...}
COHORT = Path(__file__).with_name("cohort.csv")  # 309 patients with one slide and one CT series on the IDC
PATIENTS = Path(__file__).with_name("patients.txt")  # the 168 of them behind the paper's embeddings, in row order

NUM_PATIENTS = 168
PATHOLOGY = EmbeddingFile(DATASET, "pathology", "h_optimus_0", (NUM_PATIENTS, 1536), "float32")
CT = EmbeddingFile(DATASET, "ct", "ct_fm", (NUM_PATIENTS, 512), "float32")
PAIR = (PATHOLOGY, CT)
FILES = [PATHOLOGY, CT]

# The DICOM of the 168 patients (IDC series sizes of cohort.csv) is deleted patient by patient after caching.
RAW_DATA = RawData(
    "CPTAC slides (DICOM-WSI) and CT series from the NCI Imaging Data Commons",
    download_bytes=118_572_000_000,
    stored_bytes=4_471_318_216,
)


def patients() -> list[str]:
    return PATIENTS.read_text().split()
