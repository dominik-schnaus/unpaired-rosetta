# Pairs from the CycleGAN literature (Table 11)

Five modality pairs on which CycleGAN-style translation is usually studied. Each pair embeds both modalities with one
self-supervised encoder, so the two spaces share a basis and the identity map is a baseline (Table 11). Row `i` of
the two files of a pair is the same slice, frame, patch or sentence.

| setting | files (`embeddings/<dataset>/<side>/<model>.pt`) | rows | encoder |
| --- | --- | --- | --- |
| MR ↔ CT (brain) | `SynthRAD2023-brain/{mr,ct}/dino_vit-b16@224_mean` | 3,600 | DINO ViT-B/16, mean over all tokens |
| CBCT ↔ CT (brain) | `SynthRAD2023-Task2-brain/{cbct,ct}/dinov2_vit-b14@224_mean` | 3,600 | DINOv2 ViT-B/14, mean over all tokens |
| RGB ↔ thermal | `LLVIP-train/{visible,infrared}/dinov2_vit-l14@224_mean` | 12,025 | DINOv2 ViT-L/14 |
| SAR ↔ optical | `SEN12MS-CR/{s1,s2}/dinov2_vit-l14@224_mean` | 7,899 | DINOv2 ViT-L/14 |
| speaker clb ↔ rms | `CMUArctic/{clb,rms}/wavlm-large` | 1,132 | WavLM-Large, mean over time of 4 s |

All files are float32.

## Data

`download.py` fetches every corpus into `$UNPAIRED_ROSETTA_ROOT/data/cyclegan_pairs/<corpus>/`, checks its SHA-256
and builds the caches the encoders read.

| corpus | download | on disk (archive, unpacked, cache) |
| --- | --- | --- |
| SynthRAD2023 Task 1 (Zenodo 7260705) | 14.5 GB | 29.5 GB |
| SynthRAD2023 Task 2 (Zenodo 7260705) | 10.9 GB | 22.3 GB |
| LLVIP (Hugging Face mirror of the official zip) | 4.0 GB | 8.1 GB |
| SEN12MS-CR test split (5 Parquet shards) | 2.3 GB | 5.4 GB |
| CMU Arctic, speakers clb and rms (festvox.org) | 0.18 GB | 0.43 GB |

The sizes were measured on the folders that produced the stored embeddings. SynthRAD2023 gives 20 axial slices of
each of its 180 brain patients (MR and CT scaled to [0, 1], 256 x 256 pixels, `cache/brain_<modality>.npy`). LLVIP
pairs the frames present for both cameras, and SEN12MS-CR drops the pairs whose file names disagree. CMU Arctic keeps
the prompts recorded by both speakers.

## Commands

```sh
pixi run -e modalities python -m modalities.cyclegan_pairs.download   # all corpora, or name some
pixi run -e modalities python -m modalities.cyclegan_pairs.embed      # the ten files, needs a GPU
pixi run -e modalities pytest tests/embeddings/test_cyclegan_pairs.py -s
```

The test compares the caches bit for bit and the first eight rows of every file with a cosine similarity above 0.999.
