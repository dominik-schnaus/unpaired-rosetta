# MEG ↔ text (SpanishBCBL)

The MEG response to a word read in a sentence against a sentence embedding of the same word. One session (subject
S1, block 1) of the SpanishBCBL study of Lévy et al. (2025): 244 word presentations of the reading phase cover 149
unique Spanish words. Row `i` of both files is unique word `i` in the order of its first presentation.

| side | file | shape | encoder |
| --- | --- | --- | --- |
| meg | `embeddings/SpanishBCBL-MEG-text/meg/meg_words.pt` | 149 x 306, float32 | mean over 100-500 ms after word onset, z-scored per sensor, averaged over repeats |
| text | `embeddings/SpanishBCBL-MEG-text/text/mpnet-multi.pt` | 149 x 768, float32 | `paraphrase-multilingual-mpnet-base-v2` |

Both files are L2-normalized.

## Data

The recording is distributed by the BCBL on request and must be obtained manually (1.9 GB): place
`MEG/FIF/01_9228/220404/Block1.fif` and its events table `events_S1_block1.parquet` (exported by the `neuralset`
loader of the study) below `$UNPAIRED_ROSETTA_ROOT/data/meg_text/`. `download.py` only checks that they are there.

## Commands

```sh
pixi run -e modalities python -m modalities.meg_text.download   # checks the manually obtained files
pixi run -e modalities python -m modalities.meg_text.epochs     # word epochs (MNE)
pixi run python -m modalities.meg_text.embed
pixi run -e modalities pytest tests/embeddings/test_meg_text.py -s
```

The test recomputes both files from the recording, and they equal the stored files bit for bit.
