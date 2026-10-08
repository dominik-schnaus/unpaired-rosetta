"""The raw data of the TCGA glioma pair: BraTS 2020 (manual download) and the GDC diagnostic slides of its TCGA cases.

BraTS 2020 cannot be downloaded without an account: get "BraTS 2020 training data" from Kaggle (the zip with the
folder ``BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData``, e.g. ``awsaf49/brats20-dataset-training-validation``,
4.5 GB) and place it at ``brats/BRATS-2020.zip`` under ``DATA_ROOT``. Its ``name_mapping.csv`` names the TCGA barcode
of every BraTS subject (167 of the 369).

The slides (47.5 GB for the 123 patients) are not downloaded here: ``embed.py`` fetches one at a time from the GDC,
embeds it and deletes it. ``cohort.csv`` pins the patients of the paper, the 123 of the 167 that its run embedded,
with their smallest diagnostic slide. ``--cohort`` rebuilds it for these barcodes from the BraTS mapping and the GDC
files endpoint.

    pixi run -e modalities python -m modalities.tissue_mri.download [--cohort]
"""

import argparse
import io
import json
import urllib.request
import zipfile

import pandas as pd

from modalities.tissue_mri import BRATS_FOLDER, BRATS_ZIP, COHORT, cohort

GDC = "https://api.gdc.cancer.gov"


def brats_mapping() -> pd.DataFrame:
    """BraTS subjects that are TCGA patients (``name_mapping.csv``), with their grade."""
    with zipfile.ZipFile(BRATS_ZIP) as archive:
        mapping = pd.read_csv(io.BytesIO(archive.read(f"{BRATS_FOLDER}/name_mapping.csv")))
    mapping = mapping[mapping["TCGA_TCIA_subject_ID"].astype(str).str.startswith("TCGA")]
    columns = {"BraTS_2020_subject_ID": "brats", "TCGA_TCIA_subject_ID": "barcode", "Grade": "grade"}
    return mapping.rename(columns=columns)[["barcode", "brats", "grade"]]


def diagnostic_slides(
    barcodes: list[str],
) -> dict[str, tuple[str, int]]:
    """File id and size of the smallest diagnostic slide (SVS) of every patient, from the GDC files endpoint."""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": barcodes}},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}},
        ],
    }
    query = json.dumps({"filters": filters, "fields": "file_id,file_size,cases.submitter_id", "size": 10_000}).encode()
    request = urllib.request.Request(f"{GDC}/files", data=query, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        hits = json.load(response)["data"]["hits"]
    slides = {}
    for hit in hits:
        barcode = hit["cases"][0]["submitter_id"]
        if barcode not in slides or hit["file_size"] < slides[barcode][1]:
            slides[barcode] = (hit["file_id"], int(hit["file_size"]))
    return slides


def write_cohort(
    barcodes: list[str],
) -> pd.DataFrame:
    """``cohort.csv`` for the given patients, in sorted order."""
    mapping = brats_mapping().set_index("barcode").loc[sorted(barcodes)].reset_index()
    slides = diagnostic_slides(mapping["barcode"].tolist())
    mapping["gdc_file"] = [slides[barcode][0] for barcode in mapping["barcode"]]
    mapping["gdc_size"] = [slides[barcode][1] for barcode in mapping["barcode"]]
    mapping.to_csv(COHORT, index=False)
    return mapping


def main(
    rebuild_cohort: bool,
) -> None:
    if not BRATS_ZIP.exists():
        raise SystemExit(f"Download 'BraTS 2020 training data' from Kaggle and place the zip at {BRATS_ZIP}.")
    if rebuild_cohort:
        write_cohort(cohort()["barcode"].tolist())
    patients = cohort()
    print(
        f"{len(patients)} patients, {patients['gdc_size'].sum() / 1e9:.1f} GB of slides (fetched one at a time by embed.py)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", action="store_true", help="rebuild cohort.csv from BraTS and the GDC")
    main(parser.parse_args().cohort)
