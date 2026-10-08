"""Cache the test split of the AstroCLIP DESI cross-match as two row-aligned arrays and fetch the two checkpoints.

``mhsotoudeh/astroclip`` on the Hugging Face Hub repackages the AstroCLIP cross-match. Each galaxy has a
``[152, 152, 3]`` image in the g, r and z bands (nanomaggies), a ``[7781, 1]`` spectrum, the redshift and the DESI
target id. The test split (29,697 galaxies) is streamed, not stored. Each image is stretched to RGB once with the
Legacy Survey stretch (``rgb.py``) and kept in float16.

    pixi run python -m modalities.galaxy_spectra.download
"""

import json
from collections.abc import Iterator

import numpy as np

from modalities.common import download
from modalities.galaxy_spectra import CACHE, CHECKPOINTS
from modalities.galaxy_spectra.rgb import to_rgb

REPOSITORY = "mhsotoudeh/astroclip"
REVISION = "9f36fcb2acc304c7b4fd73d452673d82ba9bf1fb"
SPLIT = "test"
IMAGE_SHAPE = (152, 152, 3)
SPECTRUM_LENGTH = 7781
CHECKPOINT_FILES = {  # name: (repository, revision, SHA-256)
    "astrodino.ckpt": (
        "polymathic-ai/astrodino",
        "7dff336ca31bdb35204c341ab6267923e8885c68",
        "0b8be1adc5f0095aba8a14e8056bc6b14e4986673a055cb37e9d3caef8c7bdf3",
    ),
    "specformer.ckpt": (
        "polymathic-ai/specformer",
        "160d67f0c07daf33d192568ca60ff38d76c39d66",
        "712c5c1fd4024baa2d85991071d5884b2b13a377bd08f94d08a403a10e9a93f8",
    ),
}


def galaxies(
    limit: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, dict[str, float]]]:
    """Yield ``(RGB image as float16, spectrum, index entry)`` for each test galaxy, in order."""
    from datasets import load_dataset

    for position, sample in enumerate(load_dataset(REPOSITORY, split=SPLIT, streaming=True, revision=REVISION)):
        if position == limit:
            return
        image = np.asarray(sample["image"], dtype=np.float32).reshape(IMAGE_SHAPE)
        spectrum = np.asarray(sample["spectrum"], dtype=np.float32).reshape(-1)
        if spectrum.shape[0] != SPECTRUM_LENGTH:
            raise ValueError(f"galaxy {position}: spectrum of length {spectrum.shape[0]}")
        entry = {"targetid": int(sample["targetid"]), "redshift": float(sample["redshift"])}
        yield to_rgb(image).astype(np.float16), spectrum, entry


def main() -> None:
    for name, (repository, revision, sha256) in CHECKPOINT_FILES.items():
        download(f"https://huggingface.co/{repository}/resolve/{revision}/{name}", CHECKPOINTS / name, sha256)
    images, spectra, index = zip(*galaxies())
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "images.npy", np.stack(images))
    np.save(CACHE / "spectra.npy", np.stack(spectra))
    (CACHE / "index.json").write_text(json.dumps(list(index)))
    print(f"cached {len(index)} galaxies in {CACHE}")


if __name__ == "__main__":
    main()
