"""Stage 0: download the single-trial NSD betas (1 mm, GLMsingle ``betas_fithrf_GLMdenoise_RR``) and keep the voxels
of the nsdgeneral region.

Replaces the authors' ``scripts/0_nsd_organize_betas.py``, which expects a full local copy of NSD. Each session
volume is downloaded, cut to the region and deleted, so the 765 GB of 1 mm betas are never on disk at once. The
output is what ``fmri_mapping.io.nsd.get_subject_roi`` reads (region 000 = all nsdgeneral voxels)::

    $NSD_DATASET/betas1mm/sub01/sub-01_roi-000.npy    [trials, voxels] int16 (beta * 300, as NSD ships them)

    pixi run -e platonic-brain python modalities/brain_scans/pipeline/extract_betas.py --subject 1
"""

import argparse
import os
import queue
import subprocess
import threading
from pathlib import Path

import nibabel
import numpy as np
from metadata import SESSIONS, TRIALS_PER_SESSION

BETAS_URL = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata"


def download_session(
    subject: int,
    session: int,
    path: Path,
) -> Path:
    url = f"{BETAS_URL}/subj{subject:02d}/func1mm/betas_fithrf_GLMdenoise_RR/betas_session{session:02d}.nii.gz"
    partial = path.with_name(path.name + ".download")
    subprocess.run(["curl", "-fsS", "--retry", "5", "--retry-delay", "10", "-o", str(partial), url], check=True)
    partial.replace(path)
    return path


def region_betas(
    volume: Path,
    mask: np.ndarray,
) -> np.ndarray:
    """The ``[trials, voxels]`` betas of one session.

    Reads the raw int16 values and skips nibabel's float64 scaling, which would take 24 GB per 1 mm session (NSD
    stores slope 1 and intercept 0).
    """
    betas = nibabel.load(volume).dataobj.get_unscaled()
    return betas[mask].T.astype(np.int16)


def extract_subject(
    subject: int,
    root: Path,
) -> None:
    output = root / "betas1mm" / f"sub{subject:02d}" / f"sub-{subject:02d}_roi-000.npy"
    if output.exists():
        return
    mask = np.asanyarray(nibabel.load(root / "masks1mm" / f"subj{subject:02d}_nsdgeneral.nii.gz").dataobj) == 1
    betas = np.empty((SESSIONS[subject] * TRIALS_PER_SESSION, int(mask.sum())), dtype=np.int16)
    downloads = root / "raw" / "func1mm" / f"subj{subject:02d}"
    downloads.mkdir(parents=True, exist_ok=True)

    # A download thread stays up to two sessions ahead, so that network and extraction overlap.
    volumes: queue.Queue[tuple[int, Path] | Exception] = queue.Queue(maxsize=2)

    def download_all() -> None:
        try:
            for session in range(1, SESSIONS[subject] + 1):
                volume = download_session(subject, session, downloads / f"betas_session{session:02d}.nii.gz")
                volumes.put((session, volume))
        except Exception as error:  # handed to the main thread, which would otherwise wait forever
            volumes.put(error)

    thread = threading.Thread(target=download_all, daemon=True)
    thread.start()
    for _ in range(SESSIONS[subject]):
        item = volumes.get()
        if isinstance(item, Exception):
            raise item
        session, volume = item
        start = (session - 1) * TRIALS_PER_SESSION
        betas[start : start + TRIALS_PER_SESSION] = region_betas(volume, mask)
        volume.unlink()
        print(f"subject {subject}: session {session} of {SESSIONS[subject]}", flush=True)
    thread.join()

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.npy")  # np.save appends ".npy" to any other suffix
    np.save(partial, betas)
    partial.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True, choices=list(SESSIONS))
    extract_subject(parser.parse_args().subject, Path(os.environ["NSD_DATASET"]))
