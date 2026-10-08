# Tissue ↔ MRI (TCGA glioma)

The diagnostic pathology slide and the pre-operative brain MRI of the same glioma patient: the TCGA cases of BraTS
2020 that have a diagnostic slide on the GDC. Row `i` of both files is patient `i` of `cohort.csv` (123 patients).

| side | file | shape | encoder |
| --- | --- | --- | --- |
| pathology | `embeddings/TCGA-glioma-path-mri/pathology/gigapath.pt` | 123 x 1536, float32 | Prov-GigaPath tile encoder, mean over 256 random tissue patches at 20x |
| mri | `embeddings/TCGA-glioma-path-mri/mri/dinov1_vitb16-comp.pt` | 123 x 768, float32 | DINO ViT-B/16, mean over five tumour slices (t1ce, t2, flair composite) |

## Data

BraTS 2020 must be downloaded manually from Kaggle ("BraTS 2020 training data", 4.5 GB) to
`$UNPAIRED_ROSETTA_ROOT/data/tissue_mri/brats/BRATS-2020.zip`. The slides (47.5 GB for the 123 patients) are fetched
from the GDC one at a time by `embed.py` and deleted after embedding. Prov-GigaPath is a gated model on the Hugging
Face Hub.

The stored slide embeddings drew their patches with an unseeded random permutation, so they cannot be recomputed
exactly, so `embed.py` seeds the draw. A new draw of the second patient agrees with the stored row at a cosine
similarity of 0.998. The MRI side is recomputed exactly.

## Commands

```sh
pixi run -e modalities python -m modalities.tissue_mri.download   # checks BraTS 2020, [--cohort] rebuilds cohort.csv
pixi run -e modalities python -m modalities.tissue_mri.embed      # needs a GPU
pixi run -e modalities pytest tests/embeddings/test_tissue_mri.py -s
```
