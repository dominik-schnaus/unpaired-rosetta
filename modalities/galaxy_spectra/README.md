# Galaxy image ↔ spectrum (AstroCLIP DESI)

AstroCLIP (Parker et al., 2024) cross-matched DESI Early Data Release spectra with DESI Legacy Imaging Survey
images. We use its test split of 29,697 galaxies. Row `i` of both embedding files is galaxy `i` of that split. The
two encoders are the single-modality models AstroCLIP starts from, each trained on one modality alone.

| side | file | shape | encoder |
| --- | --- | --- | --- |
| image | `embeddings/AstroCLIP-DESI/image/astrodino.pt` | 29,697 x 1024, float32 | AstroDINO, a DINOv2 ViT-L/12, CLS token |
| spectrum | `embeddings/AstroCLIP-DESI/spectrum/specformer.pt` | 29,697 x 768, float32 | SpecFormer, mean over positions |

Both modalities encode the redshift, so a retrieval can succeed by matching the redshift alone.
`redshift_deciles()` returns the equal-count redshift groups `z0` to `z9` of all galaxies, which Fig. 19 colours by.

## Data

`download.py` streams the test split of `mhsotoudeh/astroclip` at a fixed revision (26 Parquet shards, 9.26 GB,
not kept). It stretches every image to RGB once with the Legacy Survey stretch (`rgb.py`) and writes
`cache/images.npy` (float16, 4.12 GB), `cache/spectra.npy` (0.92 GB) and `cache/index.json` (target id and redshift).
It also downloads `pretrained/astrodino.ckpt` (1.31 GB) and `pretrained/specformer.ckpt` (0.17 GB) at fixed revisions
and checks their SHA-256. That is 10.7 GB of download and 6.5 GB on disk. The sizes are the files of the Hugging Face
repositories and of the data folder that produced the stored embeddings.

## Code

- `rgb.py`: the Legacy Survey stretch, copied from AstroCLIP (MIT license).
- `astrodino.py`: AstroDINO as a DINOv2 ViT-L from torch hub with AstroCLIP's geometry and the `teacher` weights.
- `specformer.py`: SpecFormer, copied from AstroCLIP (MIT license) without its Lightning dependency.
- `embed.py`: 144-pixel center crops in the orientation of AstroCLIP's loader, and both encoders in float32.

## Commands

```sh
pixi run python -m modalities.galaxy_spectra.download
pixi run python -m modalities.galaxy_spectra.embed       # needs a GPU
pixi run pytest tests/embeddings/test_galaxy_spectra.py -s  # recompute the first rows and compare
```
