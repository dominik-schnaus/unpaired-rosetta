"""Language <-> language: Natural Questions passages embedded by two different sentence encoders, the benchmark of
vec2vec (Jha et al., 2026) and mini-vec2vec (Dar, 2025).

The corpus is the training split of ``jxm/nq_corpus_dpr`` (5,332,023 passages, the field ``text``). Every encoder
embeds all passages in the order of the dataset, so row ``i`` of every file is the same passage. The last 8192
passages are the paired validation set. The two unpaired training sets are the first 250,000 passages of the two
halves of one random permutation of the other passages. Files: ``embeddings/nq/language/<encoder>.pt`` of shape
``[passages, 1, 768]``.
"""

from dataclasses import dataclass
from itertools import combinations

from modalities.common import EmbeddingFile, RawData

DATASET = "nq"
CORPUS = "jxm/nq_corpus_dpr"
CORPUS_REVISION = "156fa8faf1c95e5cb6fd14c499b58dfc8d149f4f"
NUM_PASSAGES = 5_332_023
HOLDOUT = 8192  # the last passages: the paired validation set
TRAIN_SIZE = 250_000  # passages per unpaired training set


@dataclass(frozen=True)
class Encoder:
    model_id: str  # sentence-transformers model on the Hugging Face Hub
    max_length: int | None  # tokens per passage, padded to this length (None: the model's limit, padded per batch)
    batch_size: int
    dtype: str  # precision of the stored file


# The stored files come from two embedding runs. gte, gtr and stella were embedded like in the vec2vec code, with 64
# tokens (padded to 64) and stored in float64. granite and e5 were embedded later with the models' own limit of 512
# tokens and stored in float32. No encoder gets a prompt (e5 would expect "passage: "). The embeddings are normalized
# to unit length when they are loaded.
ENCODERS = {
    "granite": Encoder("ibm-granite/granite-embedding-278m-multilingual", None, 59, "float32"),
    "gtr": Encoder("sentence-transformers/gtr-t5-base", 64, 1024, "float64"),
    "stella": Encoder("infgrad/stella-base-en-v2", 64, 1024, "float64"),
    "gte": Encoder("thenlper/gte-base", 64, 1024, "float64"),
    "e5": Encoder("intfloat/e5-base-v2", None, 81, "float32"),
}
PAIRS = list(combinations(ENCODERS, 2))  # the ten ordered pairs (x, y) of Table 8
DIMENSION = 768


def embedding_file(
    encoder: str,
) -> EmbeddingFile:
    return EmbeddingFile(DATASET, "language", encoder, (NUM_PASSAGES, 1, DIMENSION), ENCODERS[encoder].dtype)


RAW_DATA = {DATASET: RawData(f"{CORPUS} (train split)", download_bytes=2_220_000_000, stored_bytes=3_280_000_000)}
