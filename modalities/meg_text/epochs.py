"""One MEG vector per word presentation of the reading phase (environment ``modalities``, MNE).

The recording is restricted to its 306 MEG sensors (magnetometers and gradiometers) and band-passed to 0.5-30 Hz.
Every word onset of the reading phase opens an epoch from -100 to 600 ms, baseline-corrected on the 100 ms before the
onset and resampled to 100 Hz. The word's vector is the mean over 100-500 ms, z-scored per sensor over all words.

    pixi run -e modalities python -m modalities.meg_text.epochs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from modalities.meg_text import EVENTS, RECORDING, WORD_EPOCHS

if TYPE_CHECKING:
    import mne


def reading_words(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """The word events of the reading phase that carry a text."""
    reading = events["is_percep"].astype(str).str.lower().isin(["true", "1"])
    words = events[(events.type == "Word") & reading]
    return words[words.text.astype(str).str.strip().str.len() > 0].reset_index(drop=True)


def word_epochs(
    raw: mne.io.Raw,
    words: pd.DataFrame,
) -> tuple[torch.Tensor, pd.DataFrame]:
    """``[words, 306]`` responses (mean over 100-500 ms, z-scored per sensor) and the words that were kept."""
    import mne

    onsets = (words.start.values * raw.info["sfreq"]).astype(int) + raw.first_samp
    markers = np.c_[onsets, np.zeros(len(onsets), int), np.ones(len(onsets), int)]
    epochs = mne.Epochs(
        raw, markers, tmin=-0.1, tmax=0.6, baseline=(-0.1, 0.0), preload=True, reject=None, verbose="ERROR"
    )
    epochs.resample(100.0, verbose="ERROR")
    data = epochs.get_data(copy=True)
    pooled = data[:, :, (epochs.times >= 0.1) & (epochs.times <= 0.5)].mean(-1)
    pooled = (pooled - pooled.mean(0)) / (pooled.std(0) + 1e-9)
    return torch.tensor(pooled).float(), words.iloc[epochs.selection].reset_index(drop=True)


def load_recording() -> mne.io.Raw:
    import mne

    raw = mne.io.read_raw_fif(RECORDING, preload=True, verbose="ERROR")
    raw.pick(["mag", "grad"])
    raw.filter(0.5, 30.0, verbose="ERROR")
    return raw


if __name__ == "__main__":
    responses, words = word_epochs(load_recording(), reading_words(pd.read_parquet(EVENTS)))
    torch.save({"emb": responses, "text": words.text.tolist(), "trial_id": words.trial_id.tolist()}, WORD_EPOCHS)
    print(f"wrote {WORD_EPOCHS}: {len(words)} word epochs, {words.text.nunique()} unique words")
