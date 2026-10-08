"""Download the MassSpecGym spectra and their metadata (Bushuiev et al., 2024) from the Hugging Face Hub.

``MassSpecGym.tsv`` has one row per spectrum (identifier, SMILES, precursor, instrument, ...), ``MassSpecGym.mgf``
the same spectra in the same order. Both files are pinned to a revision of the dataset repository and checked
against their SHA-256.

    pixi run python -m modalities.msms_molecules.download
"""

from modalities.common import download
from modalities.msms_molecules import DATA_ROOT

REVISION = "d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a"
REPOSITORY = f"https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/{REVISION}"
FILES = {  # name: (path in the repository, SHA-256)
    "MassSpecGym.tsv": ("data/MassSpecGym.tsv", "0c9cc50450def3f0d4fe2dc09dea1105fc15e635db8c6656bc3e3be37a3bcd95"),
    "MassSpecGym.mgf": (
        "data/auxiliary/MassSpecGym.mgf",
        "3c3aabf929df79ad54bb5bd4ed542d139e6aad9d78ebf6198ed47d94fa72af2e",
    ),
}


def main() -> None:
    for name, (path, sha256) in FILES.items():
        print(f"{download(f'{REPOSITORY}/{path}', DATA_ROOT / name, sha256)}")


if __name__ == "__main__":
    main()
