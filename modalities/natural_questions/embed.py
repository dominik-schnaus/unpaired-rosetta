"""Embed the Natural Questions passages with one of the five sentence encoders (on a GPU, 2 to 10 hours each).

    pixi run python -m modalities.natural_questions.embed --encoder granite

The passages are embedded in the order of the dataset, in float32 and without a prompt, by the encoder's
sentence-transformers pipeline (pooling and, where the model has one, its normalization layer).
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from modalities.natural_questions import CORPUS, CORPUS_REVISION, DIMENSION, ENCODERS, NUM_PASSAGES, embedding_file

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@torch.inference_mode()
def embed_passages(
    model: SentenceTransformer,
    texts: list[str],
    max_length: int | None,
) -> torch.Tensor:
    if max_length is None:
        features = model.preprocess(texts)  # the model's own truncation, padded to the longest passage of the batch
    else:
        features = model.tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
        )
    features = {key: value.to(model.device) for key, value in features.items() if isinstance(value, torch.Tensor)}
    return model.forward(features)["sentence_embedding"]


def main(
    encoder_name: str,
) -> None:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    encoder = ENCODERS[encoder_name]
    passages = load_dataset(CORPUS, split="train", revision=CORPUS_REVISION)
    model = SentenceTransformer(encoder.model_id, device="cuda", trust_remote_code=True).float().eval()
    embeddings = torch.empty(NUM_PASSAGES, 1, DIMENSION, dtype=getattr(torch, encoder.dtype))
    for start in tqdm(range(0, NUM_PASSAGES, encoder.batch_size), desc=encoder_name):
        texts = passages[start : start + encoder.batch_size]["text"]
        embeddings[start : start + len(texts), 0] = embed_passages(model, texts, encoder.max_length).cpu()
    path = embedding_file(encoder_name).path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, choices=list(ENCODERS))
    main(parser.parse_args().encoder)
