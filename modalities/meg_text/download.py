"""The SpanishBCBL recording cannot be downloaded by a script: it must be obtained manually.

The MEG data of Levy et al. (2025) are distributed by the Basque Center on Cognition, Brain and Language (BCBL) on
request. This pair needs two files, placed below ``$UNPAIRED_ROSETTA_ROOT/data/meg_text/``:

* ``MEG/FIF/01_9228/220404/Block1.fif``: the raw recording of subject S1, block 1 (1.9 GB).
* ``events_S1_block1.parquet``: the events of that recording (one row per word, sentence and keystroke with its onset
  ``start`` in seconds, ``type``, ``text``, ``trial_id`` and ``is_percep``, which marks the reading phase), as the
  study loader ``Pinet2024Meg`` of the ``neuralset`` package exports them from the experiment logs.

The script only checks that both files are in place.

    pixi run -e modalities python -m modalities.meg_text.download
"""

from modalities.meg_text import EVENTS, RECORDING

if __name__ == "__main__":
    missing = [path for path in (RECORDING, EVENTS) if not path.exists()]
    for path in missing:
        print(f"missing {path}")
    print(
        "the SpanishBCBL files are in place" if not missing else "obtain the missing files as described in this script"
    )
