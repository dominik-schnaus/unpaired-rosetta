"""Download the NSD design files and write the two index tables the authors' ``fmri_mapping.io.nsd`` reads.

The authors built ``nsd_stimuli.csv`` and ``nsd_masks.csv`` from a full local copy of NSD. We rebuild them from the
experimental design (``nsd_expdesign.mat``, ``nsd_stim_info_merged.csv``) and check every acquired trial against the
behavioural responses of the subject.

``nsd_stimuli.csv`` has one row per *planned* trial (30,000 per subject, the full 40-session design). Subjects 3, 4, 6
and 8 stopped early, so some of their images were shown fewer than three times. The authors'
``split_repetitions(how="outer")`` needs the missing trials as rows with ``exists == False`` and no ``subject_index``.

    pixi run -e platonic-brain python modalities/brain_scans/pipeline/metadata.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

NSD_URL = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata"
TRIALS_PER_SESSION = 750
PLANNED_TRIALS = 30_000
SESSIONS = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}  # completed scan sessions per subject


def download(
    url: str,
    path: Path,
) -> Path:
    from urllib.request import urlretrieve

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, path.with_name(path.name + ".download"))
        path.with_name(path.name + ".download").replace(path)
    return path


def download_design_files(
    root: Path,
) -> None:
    """Experimental design, behavioural responses and the nsdgeneral region of every subject (1 mm)."""
    for name in ["nsd_expdesign.mat", "nsd_stim_info_merged.csv"]:
        download(f"{NSD_URL}/experiments/nsd/{name}", root / "meta" / name)
    for subject in SESSIONS:
        subject_url = f"{NSD_URL}/ppdata/subj{subject:02d}"
        download(f"{subject_url}/behav/responses.tsv", root / "behavioural" / f"responses_{subject}.tsv")
        mask = root / "masks1mm" / f"subj{subject:02d}_nsdgeneral.nii.gz"
        download(f"{subject_url}/func1mm/roi/nsdgeneral.nii.gz", mask)


def stimulus_table(
    root: Path,
) -> pd.DataFrame:
    """One row per planned trial of every subject (the module docstring explains the columns)."""
    design = scipy.io.loadmat(str(root / "meta" / "nsd_expdesign.mat"))
    master_ordering = design["masterordering"].ravel().astype(int) - 1  # trial -> one of the subject's 10,000 images
    subject_images = design["subjectim"].astype(int)  # [8, 10000] 1-based NSD ids
    shared_images = set(design["sharedix"].ravel().astype(int).tolist())
    stimulus_info = pd.read_csv(root / "meta" / "nsd_stim_info_merged.csv", usecols=["nsdId", "cocoId"])
    coco_id = dict(zip(stimulus_info["nsdId"].to_numpy() + 1, stimulus_info["cocoId"].to_numpy()))  # nsdId is 0-based
    session = np.arange(PLANNED_TRIALS) // TRIALS_PER_SESSION + 1

    tables = []
    for subject, num_sessions in SESSIONS.items():
        nsd_ids = subject_images[subject - 1, master_ordering]
        repetition = np.empty(PLANNED_TRIALS, dtype=int)  # how often the image was planned before this trial
        seen: dict[int, int] = {}
        for trial, nsd_id in enumerate(nsd_ids):
            repetition[trial] = seen.get(nsd_id, 0)
            seen[nsd_id] = repetition[trial] + 1
        exists = session <= num_sessions
        tables.append(
            pd.DataFrame(
                {
                    "subject": subject,
                    "session": session,
                    "nsd_id": nsd_ids,
                    "subject_index": np.where(exists, np.arange(PLANNED_TRIALS), np.nan),  # acquired trials come first
                    "repetition": repetition,
                    "shared": np.isin(nsd_ids, list(shared_images)),
                    "exists": exists,
                    "coco_id": [coco_id[nsd_id] for nsd_id in nsd_ids],
                    "filename": [f"raw/func1mm/subj{subject:02d}/betas_session{s:02d}.nii.gz" for s in session],
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def check_against_responses(
    table: pd.DataFrame,
    root: Path,
) -> None:
    """Every acquired trial must show the image and session that the behavioural file of the subject lists."""
    for subject in SESSIONS:
        responses = pd.read_csv(root / "behavioural" / f"responses_{subject}.tsv", sep="\t")
        acquired = table.query("subject == @subject and exists")
        same_images = np.array_equal(acquired["nsd_id"], responses["73KID"])
        if not (same_images and np.array_equal(acquired["session"], responses["SESSION"])):
            raise RuntimeError(f"The design of subject {subject} disagrees with its behavioural responses.")


def mask_table(
    root: Path,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "roi": "nsdgeneral",
                "subject": subject,
                "resolution": "func1mm",
                "mask_path": f"masks1mm/subj{subject:02d}_nsdgeneral.nii.gz",
            }
            for subject in SESSIONS
        ]
    )


def main() -> None:
    root = Path(os.environ["NSD_DATASET"])
    download_design_files(root)
    table = stimulus_table(root)
    check_against_responses(table, root)
    table.to_csv(root / "nsd_stimuli.csv", index=False)
    mask_table(root).to_csv(root / "nsd_masks.csv", index=False)
    print(f"wrote {root / 'nsd_stimuli.csv'} ({len(table)} planned trials) and {root / 'nsd_masks.csv'}")


if __name__ == "__main__":
    main()
