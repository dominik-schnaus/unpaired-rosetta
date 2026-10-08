"""The two sides of MEG <-> text: ``embeddings/SpanishBCBL-MEG-text/meg/meg_words.pt`` and ``text/mpnet-multi.pt``.

A word read several times would be its own nearest neighbour on both sides, so every unique word is one row: the
mean of its MEG vectors, and its sentence embedding. Both are L2-normalized.

    pixi run python -m modalities.meg_text.embed
"""

import torch
from torch import Tensor
from torch.nn.functional import normalize

from modalities.meg_text import MEG, TEXT, TEXT_ENCODER, WORD_EPOCHS


def mean_per_word(
    responses: Tensor,
    texts: list[str],
) -> tuple[Tensor, list[str]]:
    """The mean response of every unique word, and the words, in the order of their first presentation."""
    rows_of_word = {}
    for row, word in enumerate(str(text) for text in texts):
        rows_of_word.setdefault(word, []).append(row)
    words = list(rows_of_word)
    return torch.stack([responses[rows_of_word[word]].float().mean(0) for word in words]), words


@torch.inference_mode()
def embed_words(
    words: list[str],
    device: str = "cuda",
) -> Tensor:
    """Sentence embeddings of all words in one batch (the padding of the batch is masked by the mean pooling)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(TEXT_ENCODER, device=device)
    features = {name: value.to(device) for name, value in model.preprocess(words).items() if isinstance(value, Tensor)}
    return model.forward(features)["sentence_embedding"].cpu()


def unit_rows(
    vectors: Tensor,
) -> Tensor:
    """L2-normalized float32 rows (a NaN counts as 0)."""
    return normalize(torch.nan_to_num(vectors.float()), dim=1)


if __name__ == "__main__":
    epochs = torch.load(WORD_EPOCHS, weights_only=False, map_location="cpu")
    responses, words = mean_per_word(epochs["emb"], epochs["text"])
    for file, compute in ((MEG, lambda: responses), (TEXT, lambda: embed_words(words))):
        if file.exists():
            continue
        file.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(unit_rows(compute()), file.path)
        print(f"wrote {file.path}")
