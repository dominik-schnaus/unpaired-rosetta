"""Embed MassSpecGym: every spectrum with DreaMS, every molecule as an ECFP4 fingerprint, one row per molecule.

DreaMS (Bushuiev et al., 2025) keeps the 100 most intense peaks of a spectrum and returns the 1024-d embedding of
its contrastive head (``dreams.api.dreams_embeddings``). ECFP4 is RDKit's Morgan fingerprint of radius 2 folded to
2048 bits, stored as a float 0/1 vector (a SMILES that RDKit cannot parse gives the zero vector). Both are computed
for all spectra in file order, and the first spectrum of every molecule is kept.

Runs in the ``msms`` environment, because DreaMS pins torch 2.2 and numpy 1.25:

    pixi run -e msms python -m modalities.msms_molecules.embed
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from modalities.msms_molecules import DATA_ROOT, MOLECULE, SPECTRUM

TABLE = DATA_ROOT / "MassSpecGym.tsv"
SPECTRA = DATA_ROOT / "MassSpecGym.mgf"
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048


def smiles_of_spectra() -> list[str]:
    """The SMILES of every spectrum, after checking that the MGF lists the spectra in the order of the table."""
    table = pd.read_csv(TABLE, sep="\t")
    with open(SPECTRA) as mgf:
        identifiers = [line.strip().split("=", 1)[1] for line in mgf if line.startswith("IDENTIFIER=")]
    if identifiers != [str(identifier) for identifier in table["identifier"]]:
        raise ValueError("MassSpecGym.mgf and MassSpecGym.tsv list the spectra in different orders.")
    return table["smiles"].tolist()


def first_spectrum_per_molecule(
    smiles: list[str],
) -> np.ndarray:
    """Indices of the first spectrum of every molecule, in increasing order."""
    seen, first = set(), []
    for index, molecule in enumerate(smiles):
        if molecule not in seen:
            seen.add(molecule)
            first.append(index)
    return np.asarray(first)


def dreams_embeddings(
    mgf: Path,
) -> torch.Tensor:
    """DreaMS embeddings (``[N, 1024]``) of all spectra of an MGF file, in file order. DreaMS writes an HDF5 copy of
    the file next to it and downloads its checkpoints into its package directory on first use."""
    from dreams.api import dreams_embeddings as embed

    return torch.as_tensor(np.asarray(embed(str(mgf)))).float()


def ecfp4(
    smiles: list[str],
) -> torch.Tensor:
    """ECFP4 bit vectors (``[N, 2048]``, float 0/1)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    bits = np.zeros((len(smiles), FINGERPRINT_BITS), dtype=np.float32)
    for row, molecule in enumerate(smiles):
        parsed = Chem.MolFromSmiles(molecule) if isinstance(molecule, str) else None
        if parsed is not None:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(parsed, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS)
            bits[row, list(fingerprint.GetOnBits())] = 1.0
    return torch.from_numpy(bits)


def save(
    tensor: torch.Tensor,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    print(f"wrote {path} {tuple(tensor.shape)}")


def main() -> None:
    smiles = smiles_of_spectra()
    first = first_spectrum_per_molecule(smiles)
    print(f"{len(smiles)} spectra of {len(first)} molecules")
    if not SPECTRUM.exists():
        save(dreams_embeddings(SPECTRA)[first], SPECTRUM.path)
    if not MOLECULE.exists():
        save(ecfp4([smiles[index] for index in first]), MOLECULE.path)


if __name__ == "__main__":
    main()
