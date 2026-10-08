"""Five modality pairs of the CycleGAN literature, each embedded by one encoder on both sides (Tab. 11, App. C.4).

* MR <-> CT and CBCT <-> CT: axial brain slices of SynthRAD2023 Tasks 1 and 2, whose two volumes are registered per
  patient, so slice ``i`` shows the same anatomy in both (20 slices of each of 180 patients).
* RGB <-> thermal: the frames of LLVIP train, recorded by a visible and a thermal camera aligned in time and space.
* SAR <-> optical: co-registered Sentinel-1 radar and Sentinel-2 optical patches of the SEN12MS-CR test split.
* speaker clb <-> rms: the 1132 CMU Arctic prompts read by the two speakers clb and rms.

Row ``i`` of the two files of a pair is therefore the same slice, frame, patch or sentence. The aligners never see
this correspondence. Only the evaluation uses it.

    pixi run -e modalities python -m modalities.cyclegan_pairs.download [corpus ...]  # raw data and slice caches
    pixi run -e modalities python -m modalities.cyclegan_pairs.embed                  # the ten embedding files
"""

from modalities.common import EmbeddingFile, RawData
from unpaired_rosetta.embeddings import storage_root

DATA_ROOT = storage_root() / "data" / "cyclegan_pairs"

MR_CT_DATASET = "SynthRAD2023-brain"
CBCT_CT_DATASET = "SynthRAD2023-Task2-brain"
RGB_THERMAL_DATASET = "LLVIP-train"
SAR_OPTICAL_DATASET = "SEN12MS-CR"
SPEAKER_DATASET = "CMUArctic"

DINOV1_B16 = "dino_vit-b16@224_mean"
DINOV2_B14 = "dinov2_vit-b14@224_mean"
DINOV2_L14 = "dinov2_vit-l14@224_mean"
WAVLM_LARGE = "wavlm-large"

# The ten stored files (float32, as computed).
MR = EmbeddingFile(MR_CT_DATASET, "mr", DINOV1_B16, (3600, 768), "float32")
CT_OF_MR = EmbeddingFile(MR_CT_DATASET, "ct", DINOV1_B16, (3600, 768), "float32")
CBCT = EmbeddingFile(CBCT_CT_DATASET, "cbct", DINOV2_B14, (3600, 768), "float32")
CT_OF_CBCT = EmbeddingFile(CBCT_CT_DATASET, "ct", DINOV2_B14, (3600, 768), "float32")
VISIBLE = EmbeddingFile(RGB_THERMAL_DATASET, "visible", DINOV2_L14, (12025, 1024), "float32")
INFRARED = EmbeddingFile(RGB_THERMAL_DATASET, "infrared", DINOV2_L14, (12025, 1024), "float32")
SAR = EmbeddingFile(SAR_OPTICAL_DATASET, "s1", DINOV2_L14, (7899, 1024), "float32")
OPTICAL = EmbeddingFile(SAR_OPTICAL_DATASET, "s2", DINOV2_L14, (7899, 1024), "float32")
SPEAKER_CLB = EmbeddingFile(SPEAKER_DATASET, "clb", WAVLM_LARGE, (1132, 1024), "float32")
SPEAKER_RMS = EmbeddingFile(SPEAKER_DATASET, "rms", WAVLM_LARGE, (1132, 1024), "float32")

# The settings of Tab. 11, in its order: (space X, space Y).
PAIRS = {
    "MR ↔ CT (brain)": (MR, CT_OF_MR),
    "CBCT ↔ CT (brain)": (CBCT, CT_OF_CBCT),
    "RGB ↔ thermal": (VISIBLE, INFRARED),
    "SAR ↔ optical": (SAR, OPTICAL),
    "speaker clb ↔ rms": (SPEAKER_CLB, SPEAKER_RMS),
}
ENCODERS = {
    "MR ↔ CT (brain)": "DINOv1 ViT-B/16",
    "CBCT ↔ CT (brain)": "DINOv2 ViT-B/14",
    "RGB ↔ thermal": "DINOv2 ViT-L/14",
    "SAR ↔ optical": "DINOv2 ViT-L/14",
    "speaker clb ↔ rms": "WavLM-Large",
}
FILES = [file for pair in PAIRS.values() for file in pair]

# Measured on the archives and unpacked folders that produced the stored files. The disk keeps the archive, its
# unpacked content (for SynthRAD2023 both regions, of which the cache keeps the brain slices) and the caches.
RAW_DATA = {
    "synthrad2023": RawData(
        "SynthRAD2023 Task 1 (MR, CT) + brain slice cache",
        download_bytes=14_471_900_926,
        stored_bytes=14_471_900_926 + 14_533_026_394 + 471_859_456,
    ),
    "synthrad2023_task2": RawData(
        "SynthRAD2023 Task 2 (CBCT, CT) + brain slice cache",
        download_bytes=10_886_603_058,
        stored_bytes=10_886_603_058 + 10_939_574_987 + 471_994_938,
    ),
    "LLVIP": RawData(
        "LLVIP visible and infrared frames", download_bytes=4_003_858_172, stored_bytes=4_003_858_172 + 4_055_691_118
    ),
    "sen12mscr": RawData(
        "SEN12MS-CR test split (parquet) + patch cache",
        download_bytes=2_342_968_091,
        stored_bytes=2_342_968_091 + 3_106_615_678,
    ),
    "cmu_arctic": RawData(
        "CMU Arctic speakers clb and rms",
        download_bytes=90_892_292 + 92_541_266,
        stored_bytes=90_892_292 + 92_541_266 + 123_330_259 + 127_199_379,
    ),
}
