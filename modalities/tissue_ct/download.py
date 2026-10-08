"""Download the CPTAC slides and CT scans from the NCI Imaging Data Commons and cache what the encoders read.

Per patient of ``patients.txt`` (series of ``cohort.csv``):

* pathology: DICOM whole-slide image -> up to 196 tissue tiles of 224 x 224 pixels at about 20x (0.5 um per pixel),
  the most textured tissue blocks first -> ``[T, 3, 224, 224]`` uint8.
* CT: DICOM series -> Hounsfield units -> window [-1024, 1024] scaled to [0, 1] -> trilinear resize to
  ``[96, 224, 224]`` -> float16.

Both go to ``cache/<patient>.pt``. The DICOM (118.6 GB for the 168 patients) is deleted after every patient.

    pixi run -e modalities python -m modalities.tissue_ct.download [--limit N]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from modalities.tissue_ct import CACHE, COHORT, DATA_ROOT, patients

if TYPE_CHECKING:
    from idc_index.index import IDCClient
    from wsidicom.group import Level

SCRATCH = DATA_ROOT / "_tmp"
NUM_TILES, TILE_SIZE, TARGET_MPP = 196, 224, 0.5
CT_SHAPE = (96, 224, 224)


def is_tissue(
    pixels: np.ndarray,
) -> bool:
    """Neither background (bright) nor flat."""
    return pixels.mean() < 220 and pixels.std() > 15


def tissue_tiles(
    series_dir: Path,
) -> tuple[torch.Tensor | None, float | None]:
    """Up to ``NUM_TILES`` tissue tiles of a DICOM whole-slide image and the resolution they were read at.

    Tissue is detected on blocks of the smallest pyramid level that is at least 2500 pixels wide (the coarsest
    levels of IDC slides often read back white). The blocks are ranked by their standard deviation and read at the
    level closest to 0.5 um per pixel. If a level yields nothing among its first eight blocks, or fewer than 16 tiles
    in total, the next closest level is tried.
    """
    from wsidicom import WsiDicom

    slide = WsiDicom.open(str(series_dir))
    try:
        large = [index for index, level in enumerate(slide.levels) if level.size.width >= 2500]
        detection_index = max(large) if large else 0
        detection = slide.levels[detection_index]
        overview = np.asarray(
            slide.read_region((0, 0), detection_index, (detection.size.width, detection.size.height)).convert("RGB")
        )
        by_resolution = sorted(
            range(len(slide.levels)), key=lambda index: abs(level_mpp(slide.levels[index]) - TARGET_MPP)
        )
        for index in by_resolution:
            level = slide.levels[index]
            scale = level.size.width / detection.size.width
            block = max(int(round(TILE_SIZE / scale)), 8)
            candidates = tissue_blocks(overview, block)
            tiles = []
            for score, x, y in candidates[: NUM_TILES * 3]:
                corner = (
                    min(int(x * scale), max(level.size.width - TILE_SIZE, 0)),
                    min(int(y * scale), max(level.size.height - TILE_SIZE, 0)),
                )
                tile = np.asarray(slide.read_region(corner, index, (TILE_SIZE, TILE_SIZE)).convert("RGB"))
                if is_tissue(tile):
                    tiles.append(torch.from_numpy(tile).permute(2, 0, 1))
                if len(tiles) >= NUM_TILES:
                    break
                if not tiles and len(candidates) > 8 and score == candidates[7][0]:
                    break  # the first eight blocks gave no tissue tile at this level
            if len(tiles) >= 16:
                return torch.stack(tiles), (level.mpp.width if level.mpp else float("nan"))
        return None, None
    finally:
        slide.close()


def level_mpp(
    level: Level,
) -> float:
    return level.mpp.width if level.mpp else 0.5


def tissue_blocks(
    overview: np.ndarray,
    block: int,
) -> list[tuple[float, int, int]]:
    """``(standard deviation, x, y)`` of every tissue block of the overview, most textured first."""
    height, width = overview.shape[:2]
    blocks = []
    for y in range(0, height - block + 1, block):
        for x in range(0, width - block + 1, block):
            pixels = overview[y : y + block, x : x + block]
            if is_tissue(pixels):
                blocks.append((pixels.std(), x, y))
    return sorted(blocks, reverse=True)


def ct_volume(
    series_dir: Path,
) -> torch.Tensor | None:
    """The CT series in Hounsfield units, sorted along the patient axis, windowed and resized.

    Returns None for fewer than 20 slices.
    """
    import pydicom

    slices = []
    for file in sorted(Path(series_dir).rglob("*.dcm")):
        dataset = pydicom.dcmread(str(file))
        if not hasattr(dataset, "pixel_array"):
            continue
        position = float(getattr(dataset, "ImagePositionPatient", [0, 0, len(slices)])[2])
        hounsfield = dataset.pixel_array.astype(np.float32) * float(getattr(dataset, "RescaleSlope", 1)) + float(
            getattr(dataset, "RescaleIntercept", 0)
        )
        slices.append((position, hounsfield))
    if len(slices) < 20:
        return None
    slices.sort(key=lambda item: item[0])
    volume = np.stack([pixels for _, pixels in slices])
    volume = (np.clip(volume, -1024, 1024) + 1024) / 2048.0
    volume = torch.tensor(volume, dtype=torch.float32)[None, None]
    return F.interpolate(volume, size=CT_SHAPE, mode="trilinear", align_corners=False)[0, 0].half()


def fetch(
    client: IDCClient,
    series_uid: str,
    folder: Path,
) -> Path | None:
    """Download one DICOM series into ``folder`` and return the directory that holds its files."""
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    client.download_from_selection(downloadDir=str(folder), seriesInstanceUID=series_uid, quiet=True)
    return next((path for path in folder.rglob("*") if path.is_dir() and any(path.glob("*.dcm"))), None)


def main(
    limit: int | None = None,
) -> None:
    from idc_index import index

    cohort = pd.read_csv(COHORT)
    cohort = cohort[cohort["patient"].isin(patients())]
    cohort = cohort.assign(total_mb=cohort.ct_mb + cohort.sm_mb).sort_values("total_mb").head(limit)
    print(f"{len(cohort)} patients, {cohort.total_mb.sum() / 1000:.0f} GB of DICOM to transfer")
    CACHE.mkdir(parents=True, exist_ok=True)
    client = index.IDCClient()
    for row in cohort.itertuples():
        target = CACHE / f"{row.patient}.pt"
        if target.exists():
            continue
        try:
            slide_dir = fetch(client, row.sm_series, SCRATCH / "sm")
            tiles, mpp = tissue_tiles(slide_dir) if slide_dir else (None, None)
            ct_dir = fetch(client, row.ct_series, SCRATCH / "ct")
            volume = ct_volume(ct_dir) if ct_dir else None
            if tiles is None or volume is None:
                print(
                    f"{row.patient}: skipped (tiles={None if tiles is None else len(tiles)}, ct={volume is not None})"
                )
                continue
            torch.save({"tiles": tiles, "ct": volume, "collection": row.collection, "mpp": mpp}, target)
            print(
                f"{row.patient} {row.collection}: tiles {tuple(tiles.shape)}, ct {tuple(volume.shape)}, mpp {mpp:.2f}"
            )
        finally:
            shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only the N patients with the smallest downloads")
    main(parser.parse_args().limit)
