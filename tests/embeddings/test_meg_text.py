"""MEG and text (SpanishBCBL). Recompute both files from the recording and the words and compare them with the stored
ones.

    pixi run -e modalities pytest tests/embeddings/test_meg_text.py -s

* ``test_meg`` cuts the whole recording into epochs (MNE, a few GB of memory) and averages the responses per word for
  all 149 rows. The computation is deterministic, but the stored file was computed with MNE 1.12.1 and this
  environment has 1.13.
* ``test_text`` embeds the words with the multilingual MPNet. The stored rows were computed on another GPU, so they
  must reach a cosine similarity of 0.999, not bit equality.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest
import torch
from torch.nn.functional import cosine_similarity

from modalities import meg_text
from modalities.meg_text import embed, epochs

if TYPE_CHECKING:
    from modalities.common import EmbeddingFile

pytestmark = pytest.mark.embeddings

MIN_COSINE = 0.999


def compare(
    file: EmbeddingFile,
    computed: torch.Tensor,
) -> dict:
    if not file.exists():
        pytest.skip(f"missing {file.path}")
    stored = torch.load(file.path)
    cosines = cosine_similarity(computed, stored, dim=-1)
    numbers = {
        "file": file.relative_path,
        "rows": computed.shape[0],
        "exact": bool(torch.equal(computed, stored)),
        "max_abs_difference": (computed - stored).abs().max().item(),
        "min_cosine": cosines.min().item(),
    }
    print("RESULT", json.dumps(numbers))
    assert computed.shape == stored.shape
    return numbers


@pytest.fixture(scope="module")
def reading_words() -> pd.DataFrame:
    if not meg_text.EVENTS.exists():
        pytest.skip("the SpanishBCBL events table is missing (see download.py)")
    return epochs.reading_words(pd.read_parquet(meg_text.EVENTS))


@pytest.mark.slow
def test_meg(
    reading_words: pd.DataFrame,
) -> None:
    pytest.importorskip("mne", reason="runs in the modalities environment")
    if not meg_text.RECORDING.exists():
        pytest.skip("the SpanishBCBL recording is missing (see download.py)")
    responses, kept = epochs.word_epochs(epochs.load_recording(), reading_words)
    means, _ = embed.mean_per_word(responses, kept.text.tolist())
    assert compare(meg_text.MEG, embed.unit_rows(means))["min_cosine"] >= MIN_COSINE


def test_text(
    reading_words: pd.DataFrame,
) -> None:
    words = list(dict.fromkeys(str(text) for text in reading_words.text))  # in the order of the first presentation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert compare(meg_text.TEXT, embed.unit_rows(embed.embed_words(words, device)))["min_cosine"] >= MIN_COSINE
