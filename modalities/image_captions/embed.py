"""Compute embedding files of the paper from the raw data.

    compute_embeddings(files)                      # from Python, e.g. the notebook
    python -m modalities.image_captions.embed --dataset coco_train2014 --model mpnet [--model ...]

A vision file holds one embedding per image, ``[N, d]``. A language file holds one per text, ``[N, T, d]``. The
texts of an item are its captions (repeated for generative pooling) or the prompts of a class. Missing texts are NaN.
Items are embedded in batches, and within a batch the texts go position by position (all first captions, then all
second captions, ...), as for the stored files. Each encoder runs in the precision of its file. Finished chunks are
kept next to the file, so an interrupted run resumes where it stopped.

Generative token pooling needs vLLM, which has its own environment. From the default environment these files are
computed by ``pixi run -e vllm python -m modalities.image_captions.embed ...`` in a subprocess, one file per process.
A vLLM engine does not free its GPU memory, so each process can build only one.
"""

import argparse
import gc
import importlib.util
import shutil
import subprocess
from collections.abc import Callable

import torch
from torch import Tensor
from tqdm.auto import tqdm

from modalities import image_captions
from modalities.common import EmbeddingFile
from modalities.image_captions import datasets
from unpaired_rosetta.determinism import enable_deterministic_algorithms
from unpaired_rosetta.embeddings import storage_root

CHUNK_ITEMS = 8192  # items per resumable chunk, a multiple of each batch size
BATCH_SIZES = {  # items per forward pass (the precision decides the numbers, not the batch size)
    "dinov3_vit-7b16@512_mean": 16,
    "dinov2_vit-g14@224_mean": 64,
    "franca_vit-g14@224_laion": 64,
    "franca_vit-g14@224_laion_mean": 64,
    "qwen3-embedding-8b": 16,
    "qwen3-embedding-0.6b": 64,
}
DEFAULT_BATCH_SIZE = 128
MEAN_POOLING_TOKENS = 16384  # padded tokens per batch for mean pooling, sized for a 24 GB GPU
NUM_WORKERS = 4  # image decoding processes


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_generative(
    model: str,
) -> bool:
    return model in (image_captions.QWEN3_GEN, image_captions.QWEN3_1_7B_GEN)


class Images(torch.utils.data.Dataset):
    """Transformed images of a paired or classification dataset."""

    def __init__(
        self,
        dataset: str,
        transform: Callable,
    ) -> None:
        if dataset in datasets.CLASSIFICATION:
            self.images = datasets.classification_dataset(dataset).images
        else:
            self.images = datasets.paired_dataset(dataset)
        self.images.transform = transform

    def __len__(
        self,
    ) -> int:
        return len(self.images)

    def __getitem__(
        self,
        index: int,
    ) -> Tensor:
        return self.images[index][0]


@torch.inference_mode()
def embed_images(
    encoder: torch.nn.Module,
    images: torch.utils.data.Dataset,
    dtype: torch.dtype,
    batch_size: int,
) -> Tensor:
    """Embeddings ``[len(images), d]`` in ``dtype``. Float32 images meet ``dtype`` weights under autocast."""
    loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=NUM_WORKERS)
    embeddings = []
    for batch in tqdm(loader, desc="images", leave=False):
        with torch.autocast(device().type, dtype=dtype, enabled=dtype != torch.float32):
            embeddings.append(encoder(batch.to(device())).cpu().to(dtype))
    return torch.cat(embeddings)


@torch.inference_mode()
def embed_texts(
    encoder: Callable[[list[str]], Tensor],
    items: list[list[str]],
    dtype: torch.dtype,
    batch_size: int,
) -> Tensor:
    """Embeddings ``[len(items), T, d]`` in ``dtype``.

    T is the largest number of texts of an item, and the rest is NaN.
    """
    rows = []
    for start in tqdm(range(0, len(items), batch_size), desc="texts", leave=False):
        batch = items[start : start + batch_size]
        embedded = [[] for _ in batch]
        for position in range(max(len(texts) for texts in batch)):
            owners = [index for index, texts in enumerate(batch) if position < len(texts)]
            with torch.autocast(device().type, dtype=dtype, enabled=dtype != torch.float32):
                outputs = encoder([batch[index][position] for index in owners]).cpu().to(dtype)
            for index, output in zip(owners, outputs):
                embedded[index].append(output)
        rows.extend(embedded)
    stacked = torch.full((len(rows), max(len(row) for row in rows), rows[0][0].shape[-1]), torch.nan, dtype=dtype)
    for index, row in enumerate(rows):
        stacked[index, : len(row)] = torch.stack(row)
    return stacked


def text_items(
    dataset: str,
    model: str,
) -> list[list[str]]:
    """Texts of each row. Generative pooling wraps each text in its prompt, once per repeat."""
    items = datasets.texts(dataset)
    if not is_generative(model):
        return items
    from modalities.image_captions.generative import GENERATION_PROMPT

    repeats = image_captions.GENERATION_REPEATS.get(model, 1)
    return [[GENERATION_PROMPT.format(text) for text in texts for _ in range(repeats)] for texts in items]


def mean_pooling_batch_size(
    items: list[list[str]],
    tokenizer_name: str,
) -> int:
    """Batch size for mean pooling. A batch is padded to its longest text, and text length varies 100-fold between
    corpora. The batch size follows the longest text so that it fits a 24 GB GPU."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer([text for texts in items for text in texts], add_special_tokens=False)["input_ids"]
    longest = max(len(ids) for ids in token_ids)
    return max(1, min(32, MEAN_POOLING_TOKENS // max(longest, 1)))


def dtype_of(
    file: EmbeddingFile,
) -> torch.dtype:
    return getattr(torch, file.dtype)


def encoder_key(
    file: EmbeddingFile,
) -> tuple[str, str | None]:
    """Files with the same key share an encoder. The key is the model and the file precision. vLLM runs in the
    checkpoint precision and cannot free GPU memory for a second engine, so its key has no precision."""
    return file.model, None if is_generative(file.model) else file.dtype


def load_encoder(
    file: EmbeddingFile,
) -> tuple[Callable[..., Tensor], Callable | None, int]:
    """Encoder of a file in its precision, as ``(encoder, transform or None, batch size)``."""
    if is_generative(file.model):
        from modalities.image_captions.generative import BATCH_SIZE, GENERATIVE_ENCODERS, GenerativePooling

        checkpoint, _, memory = GENERATIVE_ENCODERS[file.model]
        return GenerativePooling(checkpoint, memory), None, BATCH_SIZE
    from modalities.image_captions import encoders

    if file.modality == "vision":
        encoder = encoders.vision_encoder(file.model, dtype_of(file), device())
        return encoder.module, encoder.transform, BATCH_SIZES.get(file.model, DEFAULT_BATCH_SIZE)
    encoder = encoders.language_encoder(file.model, dtype_of(file), device())
    return encoder, None, BATCH_SIZES.get(file.model, DEFAULT_BATCH_SIZE)


def compute_file(
    file: EmbeddingFile,
    encoder: Callable[..., Tensor],
    transform: Callable | None,
    batch_size: int,
) -> None:
    """Embed all items of ``file`` chunk by chunk and write the file. Chunks already on disk are reused."""
    parts = file.path.with_name(file.path.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    dtype = dtype_of(file)
    if file.modality == "vision":
        images = Images(file.dataset, transform)
        embed = lambda start, stop: embed_images(
            encoder, torch.utils.data.Subset(images, range(start, stop)), dtype, batch_size
        )
        num_items = len(images)
    else:
        items = text_items(file.dataset, file.model)
        if file.model.endswith("-mean"):
            batch_size = mean_pooling_batch_size(items, encoder.tokenizer.name_or_path)
        embed = lambda start, stop: embed_texts(encoder, items[start:stop], dtype, batch_size)
        num_items = len(items)
    assert num_items == file.shape[0], f"{file.relative_path}: {num_items} items, the registry expects {file.shape[0]}"
    chunks = []
    for start in range(0, num_items, CHUNK_ITEMS):
        path = parts / f"{start:09d}.pt"
        if not path.exists():
            torch.save(embed(start, min(start + CHUNK_ITEMS, num_items)), path)
        chunks.append(torch.load(path))
    write(file, chunks)
    shutil.rmtree(parts)


def write(
    file: EmbeddingFile,
    chunks: list[Tensor],
) -> None:
    """Concatenate the chunks, NaN-pad the texts to the file width and write the file atomically."""
    if file.modality != "vision":
        width = file.shape[1]
        padding = lambda chunk: chunk.new_full((chunk.shape[0], width - chunk.shape[1], chunk.shape[2]), torch.nan)
        chunks = [torch.cat([chunk, padding(chunk)], dim=1) for chunk in chunks]
    embeddings = torch.cat(chunks)
    assert tuple(embeddings.shape) == file.shape and embeddings.dtype == dtype_of(file), (file, embeddings.shape)
    partial = file.path.with_name(file.path.name + ".partial")
    torch.save(embeddings, partial)
    partial.rename(file.path)
    print(f"wrote {file.path} {tuple(embeddings.shape)} {embeddings.dtype}")


def write_labels(
    dataset: str,
) -> None:
    path = storage_root() / image_captions.label_file(dataset)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(datasets.labels(dataset), path)


def compute_embeddings(
    files: list[EmbeddingFile],
) -> None:
    """Compute the missing ``files`` and the labels of classification datasets. Each encoder is loaded once."""
    enable_deterministic_algorithms()
    missing = [file for file in dict.fromkeys(files) if not file.exists()]
    generative = [file for file in missing if is_generative(file.model)]
    if generative and importlib.util.find_spec("vllm") is None:
        compute_in_vllm_environment(generative)
        missing = [file for file in missing if file not in generative]
    for key in dict.fromkeys(encoder_key(file) for file in missing):
        batch = [file for file in missing if encoder_key(file) == key]
        encoder, transform, batch_size = load_encoder(batch[0])
        for file in batch:
            print(f"computing {file.relative_path}")
            compute_file(file, encoder, transform, batch_size)
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    for dataset in dict.fromkeys(file.dataset for file in files if file.modality == "vision"):
        if dataset in datasets.CLASSIFICATION:
            write_labels(dataset)


def compute_in_vllm_environment(
    files: list[EmbeddingFile],
) -> None:
    if shutil.which("pixi") is None:
        raise RuntimeError(
            "generative token pooling needs vLLM 0.29.0, which pins its own torch: install it into a "
            "separate environment and run python -m modalities.image_captions.embed there"
        )
    for file in files:
        subprocess.run(
            [
                "pixi",
                "run",
                "-e",
                "vllm",
                "python",
                "-m",
                "modalities.image_captions.embed",
                "--dataset",
                file.dataset,
                "--model",
                file.model,
            ],
            check=True,
        )


def registry_file(
    dataset: str,
    model: str,
) -> EmbeddingFile:
    """Registered file of ``(dataset, model)``. The model decides the modality."""
    if model in image_captions.ALL_VISION_MODELS:
        return image_captions.vision_file(dataset, model)
    return image_captions.language_file(dataset, model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", required=True, help="e.g. coco_train2014 (repeatable)")
    parser.add_argument("--model", action="append", required=True, help="e.g. dinov2_vit-b14@224_mean (repeatable)")
    arguments = parser.parse_args()
    compute_embeddings([registry_file(dataset, model) for dataset in arguments.dataset for model in arguments.model])


if __name__ == "__main__":
    main()
