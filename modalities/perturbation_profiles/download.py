"""Download the replicate-level Cell Painting and L1000 profiles of LINCS Pilot 1 from the Rosetta resource
(Haghighi et al., 2022, ``cpg0003-rosetta`` in the public Cell Painting Gallery, anonymous HTTPS).

    pixi run -e modalities python -m modalities.perturbation_profiles.download
"""

from modalities.common import download
from modalities.perturbation_profiles import REPLICATES

URL = "https://cellpainting-gallery.s3.amazonaws.com/cpg0003-rosetta/broad/workspace/curated_preprocessed_data/LINCS-Pilot1"
FILES = {  # assay: (path below URL, SHA-256)
    "morphology": (
        "CellPainting/replicate_level_cp_augmented.parquet",
        "3e41210cfe128173ee48c011f34f6215541ae1886512ce4e574852d2ce78c25f",
    ),
    "expression": (
        "L1000/replicate_level_l1k.parquet",
        "f27e3c48e7b880ed247557ab28d7e8bad9d073b78d52a8428f2c019c428baae8",
    ),
}

if __name__ == "__main__":
    for assay, (name, sha256) in FILES.items():
        print(f"wrote {download(f'{URL}/{name}', REPLICATES[assay], sha256)}")
