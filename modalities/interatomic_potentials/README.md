# MLIP ↔ MLIP (MP-20)

One crystal structure described by two machine-learning interatomic potentials. Row `i` of both files is structure
`i` of the MP-20 training split (the first 12,000 rows of CDVAE's `train.csv`).

| side | file | shape | encoder |
| --- | --- | --- | --- |
| MACE-MP-0 small | `embeddings/MP20-structure/mace_mp_small/mace_mp_small-pca256-structure.pt` | 12,000 x 256, float32 | mean over atoms of the invariant descriptors of all layers, PCA 256 |
| MACE-MP-0 medium | `embeddings/MP20-structure/mace_mp_medium/mace_mp_medium-pca256-structure.pt` | 12,000 x 256, float32 | the same |

## Data

`download.py` fetches `train.csv` of CDVAE (35 MB, CIF strings, checked by SHA-256). `descriptors.py` runs both
potentials on the GPU in the `mlip` environment (MACE 0.3.16 needs torch < 2.9) and caches `cache/structure/<potential>.npy`.
`embed.py` fits the PCA (`modalities/pca.py`) and projects in the GPU batch sizes that reproduce the stored files bit
for bit.

## Commands

```sh
pixi run -e modalities python -m modalities.interatomic_potentials.download
pixi run -e mlip python modalities/interatomic_potentials/descriptors.py
pixi run python -m modalities.interatomic_potentials.embed
pixi run pytest tests/embeddings/test_interatomic_potentials.py -s            # PCA of the cached descriptors
pixi run -e mlip pytest tests/embeddings/test_interatomic_potentials.py -s    # MACE on the first structures
```
