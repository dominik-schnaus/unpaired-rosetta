"""Download the raw data of the five pairs and build the slice and patch caches the encoders read.

    pixi run -e modalities python -m modalities.cyclegan_pairs.download [corpus ...]

The corpora are synthrad2023, synthrad2023_task2, LLVIP, sen12mscr and cmu_arctic (all by default).

* SynthRAD2023 (Zenodo 7260705, public): every brain patient with both volumes gives 20 axial slices, evenly spaced
  inside the body mask after trimming 10 % at both ends, resized to 256 x 256 by nearest neighbour. MR is scaled
  by its 1st and 99th percentile (it has no absolute unit), CT and CBCT by the fixed window [-1000, 2000] HU.
* LLVIP: a Hugging Face mirror of the official archive. The train frames are read directly.
* SEN12MS-CR: the five test shards of a Hugging Face mirror, decoded into two arrays of RGB patches.
* CMU Arctic: the archives of the speakers clb and rms from festvox.org.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from modalities.common import download
from modalities.cyclegan_pairs import DATA_ROOT

if TYPE_CHECKING:
    import pyarrow

SYNTHRAD = {  # corpus folder: (archive, task folder, modalities, url, sha256)
    "synthrad2023": (
        "SynthRAD2023_Task1.zip",
        "Task1",
        ("mr", "ct"),
        "https://zenodo.org/api/records/7260705/files/Task1.zip/content",
        "SHA_TASK1",
    ),
    "synthrad2023_task2": (
        "SynthRAD2023_Task2.zip",
        "Task2",
        ("cbct", "ct"),
        "https://zenodo.org/api/records/7260705/files/Task2.zip/content",
        "SHA_TASK2",
    ),
}
REGION = "brain"
SLICES_PER_PATIENT = 20
SLICE_SIZE = 256
CT_WINDOW = (-1000.0, 2000.0)  # the full diagnostic window, which keeps the bone contrast


def normalize_mr(
    volume: np.ndarray,
) -> np.ndarray:
    """MR has no absolute unit: scale by the 1st and 99th percentile of the non-zero voxels."""
    low, high = np.percentile(volume[volume > 0], (1, 99)) if (volume > 0).any() else (0, 1)
    return np.clip((volume - low) / max(high - low, 1e-6), 0, 1)


def normalize_ct(
    volume: np.ndarray,
) -> np.ndarray:
    """Hounsfield units are absolute, so use a fixed window.

    CBCT also measures X-ray attenuation and uses the same one.
    """
    low, high = CT_WINDOW
    return np.clip((volume - low) / (high - low), 0, 1)


NORMALIZE = {"mr": normalize_mr, "ct": normalize_ct, "cbct": normalize_ct}


def resize_to(
    image: np.ndarray,
    size: int,
) -> np.ndarray:
    """Nearest-neighbour resize by index arithmetic."""
    rows = np.linspace(0, image.shape[0] - 1, size).round().astype(int)
    columns = np.linspace(0, image.shape[1] - 1, size).round().astype(int)
    return image[rows][:, columns]


def slice_positions(
    mask: np.ndarray,
    count: int,
) -> list[int]:
    """``count`` evenly spaced axial slices inside the body mask.

    The outer 10 % at both ends (mostly air) are left out.
    """
    occupied = np.flatnonzero(mask.sum(axis=(0, 1)) > 0)
    if len(occupied) == 0:
        return []
    low, high = occupied[0], occupied[-1]
    margin = int(0.1 * (high - low))
    low, high = low + margin, high - margin
    if high <= low:
        return []
    return np.linspace(low, high, count).round().astype(int).tolist()


def synthrad_patients(
    corpus: str,
) -> list[Path]:
    """The patients of the brain region that have both volumes and a mask, sorted by id."""
    _, task, modalities, _, _ = SYNTHRAD[corpus]
    region = DATA_ROOT / corpus / task / REGION
    return sorted(
        patient
        for patient in region.iterdir()
        if patient.is_dir() and all((patient / f"{modality}.nii.gz").exists() for modality in modalities)
    )


def patient_slices(
    patient: Path,
    modalities: tuple[str, str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """The uint8 slices of one patient in both modalities and their axial positions (empty if the shapes differ)."""
    import nibabel

    load = lambda name: nibabel.as_closest_canonical(nibabel.load(patient / f"{name}.nii.gz")).get_fdata()
    volume_x, volume_y, mask = load(modalities[0]), load(modalities[1]), load("mask")
    if volume_x.shape != volume_y.shape:
        print(f"  {patient.name}: shape mismatch {volume_x.shape} vs {volume_y.shape}, skipped")
        return [], [], []
    volume_x, volume_y = NORMALIZE[modalities[0]](volume_x), NORMALIZE[modalities[1]](volume_y)
    positions = slice_positions(mask, SLICES_PER_PATIENT)
    as_image = lambda volume, position: (resize_to(volume[:, :, position], SLICE_SIZE) * 255).astype(np.uint8)
    return [as_image(volume_x, p) for p in positions], [as_image(volume_y, p) for p in positions], positions


def synthrad_cache(
    corpus: str,
    modality: str,
) -> Path:
    return DATA_ROOT / corpus / "cache" / f"{REGION}_{modality}.npy"


def prepare_synthrad(
    corpus: str,
) -> None:
    archive, task, modalities, url, sha256 = SYNTHRAD[corpus]
    if not (DATA_ROOT / corpus / task).exists():
        path = download(url, DATA_ROOT / archive, sha256)
        print(f"extracting {path}")
        with zipfile.ZipFile(path) as zipped:
            zipped.extractall(DATA_ROOT / corpus)
    if all(synthrad_cache(corpus, modality).exists() for modality in modalities):
        return
    slices_x, slices_y, index = [], [], []
    for patient in synthrad_patients(corpus):
        images_x, images_y, positions = patient_slices(patient, modalities)
        slices_x += images_x
        slices_y += images_y
        index += [{"patient": patient.name, "slice": int(position)} for position in positions]
    cache = synthrad_cache(corpus, modalities[0]).parent
    cache.mkdir(parents=True, exist_ok=True)
    np.save(synthrad_cache(corpus, modalities[0]), np.stack(slices_x))
    np.save(synthrad_cache(corpus, modalities[1]), np.stack(slices_y))
    (cache / f"{REGION}_index.json").write_text(json.dumps(index))
    print(f"{corpus}: {len(index)} slice pairs")


LLVIP_URL = "https://huggingface.co/datasets/UserNae3/LLVIP/resolve/main/LLVIP.zip"
LLVIP_SHA256 = "b3b55475c093ac54663c467859eb31a82b840b53a7196ad1887cb547e4518b45"


def llvip_frame_ids(
    split: str = "train",
) -> list[str]:
    """The frames present for both cameras, sorted. Frame ``i`` is the same instant on both sides."""
    cameras = [
        {path.stem for path in (DATA_ROOT / "LLVIP" / camera / split).glob("*.jpg")}
        for camera in ("visible", "infrared")
    ]
    return sorted(set.intersection(*cameras))


def prepare_llvip() -> None:
    if (DATA_ROOT / "LLVIP" / "visible" / "train").exists():
        return
    path = download(LLVIP_URL, DATA_ROOT / "LLVIP.zip", LLVIP_SHA256)
    print(f"extracting {path}")
    with zipfile.ZipFile(path) as zipped:
        zipped.extractall(DATA_ROOT)


SEN12MS_URL = "https://huggingface.co/datasets/mespinosami/sen12mscr/resolve/main/data/{}"
SEN12MS_SHARDS = {  # the test split
    "test-00000-of-00005-960f816cc5258e2d.parquet": "0dfb045ce5ca78733244dd72ecbc789153242637cdf2d5f4995ccf45d5a0d480",
    "test-00001-of-00005-7c2c92b4011fc8d4.parquet": "d329a627939dd1a6a12aa6e29b474bcb6fecd04934911ce545e347ee6722a477",
    "test-00002-of-00005-f5f00da1f79a99f1.parquet": "c1153e732204f3d041dd523b39b495fcb35cd79f128e30b64b477cdc0d4f5d3a",
    "test-00003-of-00005-80e71c45c7766198.parquet": "41cf1bf64190b31f871a7ee98c10ff399df42e3bdef7db1e0b3d575e979544a5",
    "test-00004-of-00005-1fd8401bd056a151.parquet": "eee5dec04d38b40b0ea5d19fffa53a9de21a71c7b6b4763dcb1cce2f8c49e348",
}


def sen12ms_cache(
    sensor: str,
) -> Path:
    return DATA_ROOT / "sen12mscr" / "cache" / f"{sensor}.npy"


def sen12ms_scene(
    path: str,
) -> str:
    """``ROIs1868_summer_s1_119_p820.png`` -> ``ROIs1868_summer_119``.

    The file names of both sensors agree on this part.
    """
    parts = Path(path).stem.split("_")
    return f"{parts[0]}_{parts[1]}_{parts[3]}"


def decode_patches(
    table: pyarrow.Table,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, str]]]:
    """The RGB patches of both sensors in one table. Pairs whose file names disagree are dropped."""
    from PIL import Image

    decode = lambda row: np.asarray(Image.open(io.BytesIO(row["bytes"])).convert("RGB"))
    sar, optical, index = [], [], []
    for row_s1, row_s2 in zip(table.column("s1").to_pylist(), table.column("s2").to_pylist()):
        if sen12ms_scene(row_s1["path"]) != sen12ms_scene(row_s2["path"]):
            continue
        sar.append(decode(row_s1))
        optical.append(decode(row_s2))
        index.append({"path": row_s1["path"], "scene": sen12ms_scene(row_s1["path"])})
    return sar, optical, index


def prepare_sen12ms() -> None:
    import pyarrow.parquet as parquet

    if sen12ms_cache("s1").exists() and sen12ms_cache("s2").exists():
        return
    sar, optical, index = [], [], []
    for name, sha256 in SEN12MS_SHARDS.items():
        shard = download(SEN12MS_URL.format(name), DATA_ROOT / "sen12mscr" / name, sha256)
        shard_sar, shard_optical, shard_index = decode_patches(parquet.ParquetFile(shard).read(columns=["s1", "s2"]))
        sar += shard_sar
        optical += shard_optical
        index += shard_index
    sen12ms_cache("s1").parent.mkdir(parents=True, exist_ok=True)
    np.save(sen12ms_cache("s1"), np.stack(sar))
    np.save(sen12ms_cache("s2"), np.stack(optical))
    (sen12ms_cache("s1").parent / "index.json").write_text(json.dumps(index))
    print(f"sen12mscr: {len(index)} patch pairs")


ARCTIC_URL = "http://festvox.org/cmu_arctic/packed/cmu_us_{}_arctic.tar.bz2"
ARCTIC_SPEAKERS = {
    "clb": "3f16dc3f3b97955ea22623efb33b444341013fc660677b2e170efdcc959fa7c6",
    "rms": "c6dc11235629c58441c071a7ba8a2d067903dfefbaabc4056d87da35b72ecda4",
}


def arctic_wav_dir(
    speaker: str,
) -> Path:
    return DATA_ROOT / "cmu_arctic" / f"cmu_us_{speaker}_arctic" / "wav"


def arctic_prompt_ids() -> list[str]:
    """The prompts recorded by both speakers, sorted. Row ``i`` is the same sentence on both sides."""
    return sorted(
        set.intersection(
            *({path.stem for path in arctic_wav_dir(speaker).glob("*.wav")} for speaker in ARCTIC_SPEAKERS)
        )
    )


def prepare_arctic() -> None:
    for speaker, sha256 in ARCTIC_SPEAKERS.items():
        if arctic_wav_dir(speaker).exists():
            continue
        path = download(
            ARCTIC_URL.format(speaker), DATA_ROOT / "cmu_arctic" / f"cmu_us_{speaker}_arctic.tar.bz2", sha256
        )
        with tarfile.open(path, "r:bz2") as archive:
            archive.extractall(DATA_ROOT / "cmu_arctic", filter="data")


PREPARE = {
    "synthrad2023": lambda: prepare_synthrad("synthrad2023"),
    "synthrad2023_task2": lambda: prepare_synthrad("synthrad2023_task2"),
    "LLVIP": prepare_llvip,
    "sen12mscr": prepare_sen12ms,
    "cmu_arctic": prepare_arctic,
}

if __name__ == "__main__":
    for corpus in sys.argv[1:] or PREPARE:
        print(f"--- {corpus}")
        PREPARE[corpus]()
