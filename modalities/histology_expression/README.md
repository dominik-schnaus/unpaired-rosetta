# Histology ↔ expression (HEST-Benchmark, CCRCC)

Each Visium spot of the 24 clear cell renal cell carcinoma slides of HEST-Benchmark is measured twice, by a
224 x 224 pixel H&E tile centred on the spot and by the RNA counts of the spot. Row `i` of both files is the same
spot (74,220 spots, joined on the barcode).

| side | file | shape | encoder |
| --- | --- | --- | --- |
| tile | `embeddings/HEST-CCRCC/tile/h_optimus_0.pt` | 74,220 x 1536, float32 | H-Optimus-0 (timm), its own normalization |
| expression | `embeddings/HEST-CCRCC/expression/expression-pca64-ccrcc.pt` | 74,220 x 64, float32 | PCA 64 of log1p counts per 10,000 of the 2,000 most variable genes |

## Data

`download.py` fetches the CCRCC folder of `MahmoodLab/hest-bench` (public): 13.9 GB of tile stores and counts.
`prepare.py` joins them into `cache/CCRCC/tiles.npy` (74,220 x 224 x 224 x 3 uint8, 11.2 GB) and
`cache/CCRCC/expression.npy` (74,220 x 2,000 float32, 0.6 GB), and the cache takes 11.8 GB. The sizes were measured on
the Hugging Face snapshot and the cache that produced the stored embeddings. H-Optimus-0 (`bioptimus/H-optimus-0`)
is a gated model: accept its terms on the Hugging Face Hub and log in first.

## Commands

```sh
pixi run -e modalities python -m modalities.histology_expression.download
pixi run -e modalities python -m modalities.histology_expression.prepare
pixi run -e modalities python -m modalities.histology_expression.embed      # needs a GPU for the tiles
pixi run -e modalities pytest tests/embeddings/test_histology_expression.py -s
```

The test checks the cache of the first slide and the PCA bit for bit (the PCA is projected in batches of 1,024 rows
on the GPU, the batch size of the stored file) and the H-Optimus-0 embeddings of the first 16 tiles by their cosine
similarity (above 0.999).
