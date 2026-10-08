"""Download the CCRCC cohort of HEST-Benchmark (``MahmoodLab/hest-bench`` on the Hugging Face Hub, public and ungated).

Per slide: ``patches/<slide>.h5`` (``img`` ``[n, 224, 224, 3]`` uint8, one tile per spot, and its ``barcode``) and
``adata/<slide>.h5ad`` (raw counts of the spots). 24 slides, 13.9 GB.

    pixi run -e modalities python -m modalities.histology_expression.download
"""

from huggingface_hub import snapshot_download

from modalities.histology_expression import COHORT, DATA_ROOT

REPOSITORY = "MahmoodLab/hest-bench"


def main() -> None:
    patterns = [f"{COHORT}/patches/*.h5", f"{COHORT}/adata/*.h5ad"]
    snapshot_download(REPOSITORY, repo_type="dataset", allow_patterns=patterns, local_dir=DATA_ROOT / "raw")


if __name__ == "__main__":
    main()
