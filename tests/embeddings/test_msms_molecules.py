"""MS/MS spectra and molecules (MassSpecGym). Recompute the first rows of both files and compare them with the stored
ones.

    pixi run -e msms pytest tests/embeddings/test_msms_molecules.py -s

* ``test_ecfp4``: the fingerprints of the first molecules must equal the stored bits exactly.
* ``test_dreams``: the spectra of the first molecules are written to a small MGF file and embedded with DreaMS. The
  stored rows were computed on another GPU, so they must reach a cosine similarity of 0.999, not bit equality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch.nn.functional import cosine_similarity

from modalities import msms_molecules

pytestmark = pytest.mark.embeddings
pytest.importorskip("dreams", reason="runs in the msms environment")
pytest.importorskip("rdkit", reason="runs in the msms environment")

from modalities.msms_molecules import embed  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

NUM_MOLECULES = 16
MIN_COSINE = 0.999
HAS_DATA = embed.TABLE.exists() and embed.SPECTRA.exists()


@pytest.fixture(scope="module")
def smiles_and_first() -> tuple[list[str], np.ndarray]:
    smiles = embed.smiles_of_spectra()
    return smiles, embed.first_spectrum_per_molecule(smiles)


def write_spectra(
    indices: list[int],
    path: Path,
) -> None:
    """Write the MGF blocks of the spectra at positions ``indices`` in the file, in that order."""
    wanted, blocks, block, position = set(indices), {}, [], 0
    with open(embed.SPECTRA) as mgf:
        for line in mgf:
            block.append(line)
            if line.startswith("END IONS"):
                if position in wanted:
                    blocks[position] = "".join(block).strip()
                position, block = position + 1, []
                if len(blocks) == len(wanted):
                    break
    path.write_text("\n\n".join(blocks[index] for index in indices) + "\n")


@pytest.mark.skipif(not HAS_DATA, reason="MassSpecGym is not downloaded")
def test_ecfp4(
    smiles_and_first: tuple[list[str], np.ndarray],
) -> None:
    smiles, first = smiles_and_first
    computed = embed.ecfp4([smiles[index] for index in first[:NUM_MOLECULES]])
    stored = torch.load(str(msms_molecules.MOLECULE.path), mmap=True)[:NUM_MOLECULES]
    print(f"RESULT {msms_molecules.MOLECULE.relative_path}: {int((computed != stored).sum())} differing bits")
    assert torch.equal(computed, stored)


@pytest.mark.slow
@pytest.mark.skipif(not HAS_DATA, reason="MassSpecGym is not downloaded")
def test_dreams(
    smiles_and_first: tuple[list[str], np.ndarray],
    tmp_path: Path,
) -> None:
    _, first = smiles_and_first
    subset = tmp_path / "spectra.mgf"
    write_spectra(first[:NUM_MOLECULES].tolist(), subset)
    computed = embed.dreams_embeddings(subset)
    stored = torch.load(str(msms_molecules.SPECTRUM.path), mmap=True)[:NUM_MOLECULES]
    cosines = cosine_similarity(computed, stored, dim=-1)
    print(
        f"RESULT {msms_molecules.SPECTRUM.relative_path}: min cosine {cosines.min().item():.7f}, "
        f"max |diff| {(computed - stored).abs().max().item():.3g}"
    )
    assert cosines.min() > MIN_COSINE
