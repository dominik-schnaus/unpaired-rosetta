# Cell Painting ↔ L1000 (Rosetta, LINCS Pilot 1)

The morphology and the transcription of A549 cells under the same perturbation. Row `i` of both files is the same
(compound, dose) unit: its consensus Cell Painting profile against its consensus L1000 profile. The two assays run on
separate plates, so the sides share the perturbation and nothing else.

| side | file | shape | encoder |
| --- | --- | --- | --- |
| morphology | `embeddings/Rosetta-LINCS-Pilot1/morphology/morphology-pca512.pt` | 2,732 x 512, float32 | PCA 512 of 1,783 CellProfiler features |
| expression | `embeddings/Rosetta-LINCS-Pilot1/expression/expression-pca256.pt` | 2,732 x 256, float32 | PCA 256 of the 978 L1000 landmark genes |

## Data

`download.py` fetches the replicate-level profiles of both assays from the Cell Painting Gallery (`cpg0003-rosetta`,
anonymous HTTPS, 475 MB, checked by SHA-256). `profiles.py` keeps the 2,732 units that are reproducible in both assays
and takes the median over their replicates. `embed.py` fits the PCA (`modalities/pca.py`).

## Commands

```sh
pixi run -e modalities python -m modalities.perturbation_profiles.download
pixi run -e modalities python -m modalities.perturbation_profiles.profiles
pixi run python -m modalities.perturbation_profiles.embed
pixi run -e modalities pytest tests/embeddings/test_perturbation_profiles.py -s
```

The test runs the whole chain, and on an RTX 4500 Ada it equals the stored files bit for bit.
