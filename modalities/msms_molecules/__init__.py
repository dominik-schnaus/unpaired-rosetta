"""MS/MS spectra <-> molecules (MassSpecGym): a tandem mass spectrum and the structure of the molecule it measures.

MassSpecGym holds 231,104 spectra of 31,602 molecules. Every molecule keeps one spectrum, the first in file order,
so row ``i`` of both files is molecule ``i`` (the order of first appearance). Otherwise the roughly seven spectra
of a molecule would be each other's nearest neighbours. The spectra are embedded with DreaMS, a transformer
trained self-supervised on unannotated spectra, and the molecules with ECFP4 fingerprints (radius 2, 2048 bits),
so the two sides share nothing but the chemistry.

    pixi run python -m modalities.msms_molecules.download          # MassSpecGym.tsv and MassSpecGym.mgf
    pixi run -e msms python -m modalities.msms_molecules.embed     # embeddings/MassSpecGym/{spectrum,molecule}/
"""

from modalities.common import EmbeddingFile, RawData
from unpaired_rosetta.embeddings import storage_root

DATASET = "MassSpecGym"
DATA_ROOT = storage_root() / "data" / "msms_molecules"  # MassSpecGym.tsv, MassSpecGym.mgf (+ DreaMS's .hdf5 of it)

NUM_MOLECULES = 31_602
SPECTRUM = EmbeddingFile(DATASET, "spectrum", "dreams", (NUM_MOLECULES, 1024), "float32")
MOLECULE = EmbeddingFile(DATASET, "molecule", "ecfp4", (NUM_MOLECULES, 2048), "float32")
PAIR = (SPECTRUM, MOLECULE)
FILES = [SPECTRUM, MOLECULE]

# The two downloaded files. On first use DreaMS also writes a 594 MB HDF5 copy of the MGF next to it and fetches
# its two checkpoints (2.63 GB) into its package directory.
RAW_DATA = RawData(
    "MassSpecGym spectra (MGF) and metadata (TSV)", download_bytes=571_473_488, stored_bytes=1_165_485_024
)
