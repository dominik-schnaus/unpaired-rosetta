"""MEG <-> text (SpanishBCBL): the brain response to a word read in a sentence, and a sentence embedding of that word.

One session (subject S1, block 1) of the SpanishBCBL MEG study of Levy et al. (2025): 244 word presentations of the
reading phase cover 149 unique Spanish words. The MEG side is the 306-sensor response averaged over 100-500 ms after
word onset, z-scored per sensor and averaged over the repeats of a word. The text side is the word embedded by
``paraphrase-multilingual-mpnet-base-v2``. Row ``i`` of both files is unique word ``i`` in the order of its first
presentation. Both sides are L2-normalized.

    pixi run -e modalities python -m modalities.meg_text.download   # checks the manually obtained recording
    pixi run -e modalities python -m modalities.meg_text.epochs     # word epochs of the recording (MNE)
    pixi run python -m modalities.meg_text.embed                    # embeddings/SpanishBCBL-MEG-text/{meg,text}/
"""

from modalities.common import EmbeddingFile, RawData
from unpaired_rosetta.embeddings import storage_root

DATASET = "SpanishBCBL-MEG-text"
DATA_ROOT = storage_root() / "data" / "meg_text"
TAG = "S1_block1"
RECORDING = DATA_ROOT / "MEG" / "FIF" / "01_9228" / "220404" / "Block1.fif"  # subject S1, block 1
EVENTS = DATA_ROOT / f"events_{TAG}.parquet"  # word and sentence onsets of the recording, with their text
WORD_EPOCHS = DATA_ROOT / f"brain_words_{TAG}.pt"  # written by epochs.py: {"emb": [244, 306], "text", "trial_id"}
TEXT_ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

NUM_WORDS = 149
MEG = EmbeddingFile(DATASET, "meg", "meg_words", (NUM_WORDS, 306), "float32")
TEXT = EmbeddingFile(DATASET, "text", "mpnet-multi", (NUM_WORDS, 768), "float32")
PAIR = (MEG, TEXT)
FILES = [MEG, TEXT]

RAW_DATA = RawData(
    "SpanishBCBL MEG recording of subject S1, block 1 (FIF), and its events table",
    download_bytes=1_898_311_075,
    stored_bytes=1_898_311_075,
)
