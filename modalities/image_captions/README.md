# Vision-language embeddings

This package computes every vision-language embedding file of the paper from the public raw data. The notebook and
the experiments only read the files, so you need this package only if you do not download the precomputed files
from the Hugging Face Hub (`modalities.common.download_precomputed`).

| File | Content |
| --- | --- |
| `__init__.py` | The registry: dataset and model names (= file names), shapes and precision of the stored files, the files of every experiment (`paper_files()`), raw data sizes |
| `datasets.py` | Downloads the raw data and lists every corpus in the row order of the stored files |
| `encoders.py` | The vision and language encoders with their preprocessing and pooling |
| `generative.py` | Generative token pooling with vLLM (separate environment) |
| `embed.py` | `compute_embeddings(files)` and the command line |
| `upload_to_hub.py` | Publishes the stored files on the Hub (maintainers only) |

## Computing embeddings

```bash
pixi run python -m modalities.image_captions.embed --dataset coco_train2014 --model dinov2_vit-b14@224_mean --model mpnet
pixi run -e vllm python -m modalities.image_captions.embed --dataset coco_train2014 --model qwen3-8b-gen_69445574b3021b8a2733e87d8603d769
```

From Python, `compute_embeddings(files)` takes `EmbeddingFile`s of the registry (for example
`image_captions.files_of_setting(...)` or `image_captions.paper_files()`). It downloads the raw data it needs, loads
each encoder once, and writes `$UNPAIRED_ROSETTA_ROOT/embeddings/<dataset>/<modality>/<model>.pt`. Finished chunks of
8192 items are kept in `<file>.parts/`, so an interrupted run resumes. Files of generative token pooling are computed
in the `vllm` environment through `pixi run -e vllm ...` when vLLM is not importable.

Every encoder runs in the precision of its file: bfloat16 weights under bfloat16 autocast for the bfloat16 files,
float32 throughout for the few files that were stored in float32 (`FLOAT32_FILES`). Images and texts are embedded in
batches, and the texts of a batch position by position (all first captions, then all second captions). Language files
are `[items, texts, d]` with NaN for missing texts: MS COCO has up to 7 captions per image, the CIFAR files hold the
18 CLIP templates of every class, and generative pooling repeats each prompt three times.

## Environments

* `default` (python 3.14, torch 2.14, transformers 5.12, sentence-transformers 5.7, timm 1.0.29): every vision encoder
  and every language encoder except generative token pooling (the RAE encoder uses `diffusers` 0.38).
* `vllm` (python 3.12, vLLM 0.29.0, torch 2.13, transformers 5.17, CUDA toolkit 13.0 for FlashInfer's JIT kernels):
  generative token pooling. vLLM pins its own torch, so it cannot share the default environment. Run one generative
  model per process (a vLLM engine keeps its GPU memory until the process ends).

## Reproducibility

Every recomputed row agrees with the stored file to a cosine similarity above 0.999 (most above 0.9999), except for
generative token pooling. Different GPUs, kernels and batch sizes change the last bits, not the embedding.
Generative pooling decodes greedily, so a last-bit difference can switch one generated token and from then on the
continuation. The stored 8B files were computed on the cluster GPUs, with vLLM's prefix caching on (which makes a call
depend on what the engine computed before). Our engine turns prefix caching off and is bit-reproducible run to run on
the same GPU, but on another GPU a few texts decode differently: on the RTX 4500 the mean cosine similarity per text
is 0.980 to 0.996 and single texts go down to 0.82 (short CIFAR prompts). The 1.7B ladder files were computed on the
same RTX 4500, and six of its thirteen rungs are reproduced bit for bit.

The torch.hub code of DINOv2, DINOv3, iBOT and Franca is pinned to fixed commits (`encoders.py`). The Hugging Face
checkpoints (`all-mpnet-base-v2`, `facebook/contriever`, `Qwen/Qwen3-Embedding-{0.6B,8B}`, `Qwen/Qwen3-{1.7B,8B}`,
`nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08`) are downloaded into the Hugging Face cache on first use.

**DINOv3 weights are gated.** Request access on
<https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>, and Meta sends an e-mail with download links.
Download `dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth` (26.9 GB) into `$UNPAIRED_ROSETTA_ROOT/weights/`, or set
`DINOV3_WEIGHTS` to its path, or set `DINOV3_WEIGHTS_URL` to its link and let the code download it.

## Raw data

Raw data goes to `$UNPAIRED_ROSETTA_ROOT/raw` (`$UNPAIRED_ROSETTA_RAW` overrides it). Only the files that the
embeddings need are unpacked and every archive is deleted after unpacking. The sizes were measured on the archives and
on the unpacked files (`du --apparent-size`), with 1 GB = 10^9 bytes.

| Dataset | Source | Download | Stored |
| --- | --- | --- | --- |
| MS COCO 2014 train | images.cocodataset.org (`train2014.zip`, `annotations_trainval2014.zip`) | 13.76 GB | 13.60 GB |
| MS COCO 2014 val | images.cocodataset.org (`val2014.zip`, annotations) | 6.90 GB | 6.74 GB |
| Stanford Paragraph Captioning | Visual Genome `images.zip`, `images2.zip`, `paragraphs_v1.json.zip` | 15.21 GB | 2.60 GB |
| Densely Captioned Images | `dci.tar.gz` (dl.fbaipublicfiles.com) and SA-1B `sa_000138.tar` | 12.22 GB | 8.90 GB |
| DOCCI | storage.googleapis.com/docci | 7.60 GB | 7.64 GB |
| CIFAR-10 / CIFAR-100 | cs.toronto.edu via torchvision | 0.17 GB each | 0.19 GB each |
| ImageNet-100 | ImageNet-1k `ILSVRC2012_img_val.tar` + devkit (image-net.org) | 6.75 GB | 0.69 GB |
| WIT, first 5120 pairs | `wikimedia/wit_base` on the Hub (first row group of the first shard) | 0.47 GB | 0.16 GB |
| CC12M, first 5120 pairs | first img2dataset shard of `cc12m.tsv` | 0.30 GB | 0.18 GB |

Text-only files (the language files of the corpora and of the caption ladder) need only the captions: the COCO
annotations (0.25 GB), the SPC paragraphs (2 MB), the DCI annotations (0.87 GB) or the DOCCI descriptions (11 MB).
The class prompts of CIFAR and ImageNet-100 are part of the code.

Three sources need a manual step:

* **ImageNet-100** is cut from the ImageNet-1k validation set, which image-net.org hands out after registration. Put
  `ILSVRC2012_img_val.tar` and `ILSVRC2012_devkit_t12.tar.gz` into `raw/imagenet/`. The code keeps the 5000 images of
  the 100 classes of `IN100.txt` (Tian et al., 2020) in `raw/imagenet100/val/`.
* **DCI photos** are the SA-1B tar `sa_000138.tar`. Request the SA-1B links on
  <https://ai.meta.com/datasets/segment-anything-downloads/>, and set `DCI_IMAGES_URL` to the link of
  `sa_000138.tar` (or put the tar into `raw/densely_captioned_images/`).
* **CC12M** is a list of image URLs. The paper used the first shard of an img2dataset crawl of `cc12m.tsv` from
  2024-05-03 (`img2dataset --url_list cc12m.tsv --input_format tsv --url_col url --caption_col caption
  --output_format webdataset --image_size 336 --resize_mode border --number_sample_per_shard 10000`), read in tar order.
  About 30 % of the URLs were dead already then, and the crawl writes the samples in the order in which they arrive, so
  a new crawl gives other pairs. Put its shards into `raw/cc12m/` to compute new embeddings. The embeddings of the
  paper can only be downloaded.

The WIT subset streamed from the Hub is byte for byte the one of the paper (checked against the local copy of the
dataset the paper used).

## Duration

Extrapolated from timing 512 items (128 for generative pooling) on one RTX 4500 Ada (24 GB) with 4 image-decoding
workers. The grid has 270,500 images (MS COCO train and val, SPC, DCI, DOCCI, CIFAR-10/100, ImageNet-100).

| Encoder | Time per item | The whole grid |
| --- | --- | --- |
| DINOv3 ViT-7B/16 at 512 px | 0.22 s | 16 h |
| Franca ViT-G/14 (CLS or mean, float32 for most files) | 0.048 s | 3.6 h each |
| DINOv2 ViT-G/14 | 0.014 s | 1.1 h |
| DINOv2 ViT-B/14, iBOT ViT-B/16 and Swin-T (bound by image decoding) | 0.008 s | 0.6 h each |
| RAE encoder (MS COCO train only) | 0.010 s | 13 min |
| MPNet, Contriever | 0.002 to 0.003 s | a few minutes |
| Qwen3-Embedding-8B | 0.022 s (MS COCO, 5 captions) to 0.047 s (DOCCI) | 1.3 h |
| Qwen3-8B generative pooling (vLLM) | 1.4 s (MS COCO, 5 captions x 3 repeats) to 0.36 s (DOCCI) | about 52 h, 48 h of it for MS COCO |
| Qwen3-8B mean pooling (5120 items per corpus) | 0.024 s (MS COCO) to 0.055 s (DOCCI) | 15 min |

The other token-pooling files hold 5120 items each and take minutes, except generative pooling of App. D (about 2 h
for the six corpora).

## Tests

`tests/embeddings/test_image_captions.py` recomputes the first rows of every file of `paper_files()` and compares them
with the stored files (`pixi run pytest -m embeddings tests/embeddings/test_image_captions.py -s` in the default
environment, and with `-e vllm` for generative token pooling).
