# MS/MS spectra ↔ molecules (MassSpecGym)

MassSpecGym (Bushuiev et al., 2024) holds 231,104 tandem mass spectra of 31,602 molecules. We keep the first
spectrum of every molecule in file order, so row `i` of both embedding files is molecule `i` in the order of its first
appearance. The spectra are embedded with DreaMS and the molecules as ECFP4 fingerprints.

| side | file | shape | encoder |
| --- | --- | --- | --- |
| spectrum | `embeddings/MassSpecGym/spectrum/dreams.pt` | 31,602 x 1024, float32 | DreaMS embedding model (100 most intense peaks) |
| molecule | `embeddings/MassSpecGym/molecule/ecfp4.pt` | 31,602 x 2048, float32 | RDKit Morgan fingerprint, radius 2, 2048 bits |

## Data

`download.py` fetches `MassSpecGym.tsv` (262.3 MB) and `MassSpecGym.mgf` (309.1 MB) from the dataset repository
`roman-bushuiev/MassSpecGym` at a fixed revision and checks their SHA-256. That is 571.5 MB of download. On the first
run, DreaMS writes an HDF5 copy of the MGF file next to it (594.0 MB) and downloads its two checkpoints
(`embedding_model.ckpt` and `ssl_model.ckpt`, 2.63 GB together) into its own package directory. The sizes are the
files of the dataset repository and of the data folder that produced the stored embeddings.

## Commands

```sh
pixi run python -m modalities.msms_molecules.download
pixi run -e msms python -m modalities.msms_molecules.embed
pixi run -e msms pytest tests/embeddings/test_msms_molecules.py -s   # recompute the first rows and compare
```

The embedding runs in the `msms` environment, because DreaMS pins torch 2.2.1 and numpy 1.25. RDKit is the one
DreaMS installs, the same as for the stored files. DreaMS uses the GPU if there is one.
