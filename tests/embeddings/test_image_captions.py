"""Recompute the first rows of each vision-language embedding file of the paper and compare them with the stored file.

    pixi run pytest -m embeddings tests/embeddings -s              # every encoder but generative token pooling
    pixi run -e vllm pytest -m embeddings tests/embeddings -s      # generative token pooling (vLLM)

The stored files were computed on other GPUs, in other batches and with other kernels (for example xformers
attention). The recomputed rows therefore agree up to bfloat16 rounding, not bit for bit. Each text and image must
reach a cosine similarity of at least 0.999 with its stored row.

Generative pooling cannot meet this. The stored files were computed with the prefix caching of vLLM, which makes the
result of a call depend on what the engine computed before. Greedy decoding then turns a last-bit difference into a
different continuation for a few texts. Our engine runs without prefix caching and gives the same result on every run.
Where the cache state happens to agree (for example ``DOCCIDataset_ladder-keywords_12``), the rows are bit-identical
to the stored ones. The test embeds a whole vLLM batch (64 items) and requires a mean cosine similarity above 0.975.
The worst stored files, the class prompts of CIFAR and ImageNet-100 with Qwen3-8B, reach 0.980.
Each test prints one ``RESULT`` line with the numbers.
"""

import gc
import importlib.util
import json

import pytest
import torch
from torch.nn.functional import cosine_similarity

from modalities import image_captions
from modalities.image_captions import datasets, embed
from unpaired_rosetta.determinism import enable_deterministic_algorithms
from unpaired_rosetta.embeddings import reduce_texts, storage_root

pytestmark = pytest.mark.embeddings
enable_deterministic_algorithms()  # as in compute_embeddings

NUM_ITEMS = 8
NUM_GENERATIVE_ITEMS = 64  # one vLLM call per text position, as in the stored files
MIN_COSINE = 0.999
MIN_GENERATIVE_MEAN_COSINE = 0.975
FILES = sorted(image_captions.paper_files(), key=lambda file: (str(embed.encoder_key(file)), file.dataset))
HAS_VLLM = importlib.util.find_spec("vllm") is not None


def stored_rows(
    file: image_captions.EmbeddingFile,
    count: int,
) -> torch.Tensor:
    source = image_captions.SLICED_FROM.get((file.dataset, file.modality, file.model), file.relative_path)
    return torch.load(storage_root() / source, mmap=True)[:count]


_loaded: dict[object, tuple[object, object, int]] = {}


def encoder_of(
    file: image_captions.EmbeddingFile,
) -> tuple[object, object, int]:
    """Keep one encoder loaded at a time. The files are sorted by model, so each encoder is loaded once."""
    key = embed.encoder_key(file)
    if key not in _loaded:
        _loaded.clear()
        gc.collect()  # a vLLM engine holds its GPU memory until it is collected
        torch.cuda.empty_cache()
        _loaded[key] = embed.load_encoder(file)
    return _loaded[key]


def recompute(
    file: image_captions.EmbeddingFile,
) -> torch.Tensor:
    encoder, transform, batch_size = encoder_of(file)
    dtype = embed.dtype_of(file)
    if file.modality == "vision":
        images = torch.utils.data.Subset(embed.Images(file.dataset, transform), range(NUM_ITEMS))
        return embed.embed_images(encoder, images, dtype, NUM_ITEMS)
    count = NUM_GENERATIVE_ITEMS if embed.is_generative(file.model) else NUM_ITEMS
    items = embed.text_items(file.dataset, file.model)[:count]
    return embed.embed_texts(encoder, items, dtype, min(batch_size, count))


def compare(
    file: image_captions.EmbeddingFile,
    computed: torch.Tensor,
) -> dict:
    stored = stored_rows(file, computed.shape[0])
    if file.modality != "vision":
        width = stored.shape[1] - computed.shape[1]
        padding = computed.new_full((computed.shape[0], width, computed.shape[2]), torch.nan)
        computed = torch.cat([computed, padding], dim=1)
        assert torch.equal(computed.isnan().all(-1), stored.isnan().all(-1)), "different numbers of texts per item"
    assert computed.dtype == stored.dtype
    valid = ~stored.isnan().any(-1)
    computed_rows, stored_rows_ = computed[valid].float(), stored[valid].float()
    cosines = cosine_similarity(computed_rows, stored_rows_, dim=-1)
    item_cosines = cosine_similarity(reduce_texts(computed).float(), reduce_texts(stored).float(), dim=-1)
    return {
        "file": file.relative_path,
        "items": computed.shape[0],
        "texts": int(valid.sum()),
        "exact": bool(torch.equal(computed_rows, stored_rows_)),
        "max_abs_difference": (computed_rows - stored_rows_).abs().max().item(),
        "min_cosine": cosines.min().item(),
        "mean_cosine": cosines.mean().item(),
        "min_item_cosine": item_cosines.min().item(),
    }


@pytest.mark.gpu
@pytest.mark.parametrize("file", FILES, ids=lambda file: file.relative_path)
def test_recomputed_rows_match_stored_rows(
    file: image_captions.EmbeddingFile,
) -> None:
    if embed.is_generative(file.model) != HAS_VLLM:
        pytest.skip("generative token pooling runs in the vllm environment, everything else in the default one")
    result = compare(file, recompute(file))
    print("RESULT", json.dumps(result))
    if embed.is_generative(file.model):
        assert result["mean_cosine"] > MIN_GENERATIVE_MEAN_COSINE
    else:
        assert result["min_cosine"] > MIN_COSINE


@pytest.mark.parametrize("dataset", image_captions.CLASSIFICATION_DATASETS)
def test_labels_match_stored_labels(
    dataset: str,
) -> None:
    stored = torch.load(storage_root() / image_captions.label_file(dataset))
    assert torch.equal(datasets.labels(dataset), stored)


@pytest.mark.parametrize("file", FILES, ids=lambda file: file.relative_path)
def test_registry_describes_stored_file(
    file: image_captions.EmbeddingFile,
) -> None:
    source = image_captions.SLICED_FROM.get((file.dataset, file.modality, file.model))
    stored = torch.load(storage_root() / (source or file.relative_path), mmap=True)
    assert tuple(stored.shape[1:]) == file.shape[1:] and str(stored.dtype) == f"torch.{file.dtype}"
    assert stored.shape[0] == file.shape[0] or (source is not None and stored.shape[0] > file.shape[0])


def test_file_names_hash_the_prompts() -> None:
    """The suffix of a generative model and of the CIFAR language folder is the md5 of the prompt list."""
    import hashlib

    def md5(
        prompts: list[str],
    ) -> str:
        text = json.dumps(prompts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.md5(text.encode()).hexdigest()

    from modalities.image_captions.generative import GENERATION_PROMPT

    assert image_captions.QWEN3_GEN.endswith(md5([GENERATION_PROMPT] * 3))
    assert image_captions.QWEN3_1_7B_GEN.endswith(md5([GENERATION_PROMPT]))
    assert image_captions.CLASS_PROMPT_FOLDERS["CIFAR-10"] == f"language_{md5(datasets.CIFAR_TEMPLATES)}"
