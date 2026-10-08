"""MACE descriptors of the MP-20 structures: the forward pass of both potentials (environment ``mlip``, GPU).

Every potential reads the same 12,000 structures in the same order. For every atom, MACE returns its invariant
features of all interaction layers (256 for MACE-MP-0 small and medium). A structure is described by the mean over
its atoms. The descriptors are cached as ``<potential>.npy`` next to ``index.json`` (material id and number of atoms
of every row).

The ``mlip`` environment has MACE but not the packages of this repository. The script therefore imports nothing
from the repository and repeats the paths of ``modalities/interatomic_potentials/__init__.py``.

    pixi run -e mlip python modalities/interatomic_potentials/descriptors.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import Structure

DATA_ROOT = Path(os.environ.get("UNPAIRED_ROSETTA_ROOT", "storage")).expanduser() / "data" / "interatomic_potentials"
STRUCTURES_CSV = DATA_ROOT / "raw" / "train.csv"
DESCRIPTORS = DATA_ROOT / "cache" / "structure"
NUM_STRUCTURES = 12_000
POTENTIALS = {"mace_mp_small": "small", "mace_mp_medium": "medium"}  # name: model of mace.calculators.mace_mp


def load_structures(
    limit: int = NUM_STRUCTURES,
) -> tuple[list[Structure], list[str]]:
    """The first ``limit`` structures of ``train.csv`` as pymatgen structures, and their material ids."""
    import pandas as pd
    from pymatgen.core import Structure

    frame = pd.read_csv(STRUCTURES_CSV).head(limit)
    structures = [Structure.from_str(cif, fmt="cif") for cif in frame["cif"]]
    return structures, [str(material_id) for material_id in frame["material_id"]]


def structure_descriptors(
    model: str,
    structures: list[Structure],
    device: str = "cuda",
) -> np.ndarray:
    """``[structures, 256]``: the mean over the atoms of MACE's invariant descriptors of all layers."""
    from mace.calculators import mace_mp
    from pymatgen.io.ase import AseAtomsAdaptor

    calculator = mace_mp(model=model, device=device, default_dtype="float32")
    means = []
    for structure in structures:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        per_atom = np.asarray(calculator.get_descriptors(atoms, invariants_only=True, num_layers=-1))
        means.append(per_atom.mean(0))
    return np.stack(means).astype(np.float32)


def main() -> None:
    structures, material_ids = load_structures()
    DESCRIPTORS.mkdir(parents=True, exist_ok=True)
    if not (DESCRIPTORS / "index.json").exists():
        index = [{"structure_id": id_, "num_atoms": len(structure)} for id_, structure in zip(material_ids, structures)]
        (DESCRIPTORS / "index.json").write_text(json.dumps(index))
    for potential, model in POTENTIALS.items():
        path = DESCRIPTORS / f"{potential}.npy"
        if path.exists():
            continue
        descriptors = structure_descriptors(model, structures)
        np.save(path, descriptors)
        print(f"wrote {path} {descriptors.shape}")


if __name__ == "__main__":
    main()
