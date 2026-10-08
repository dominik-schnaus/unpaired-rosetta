"""The Legacy Survey stretch from fluxes (nanomaggies in the g, r and z bands) to an RGB image in [0, 1].

Copied from AstroCLIP (``astroclip/astrodino/data/augmentations.py``, MIT license), which copied it from
https://github.com/legacysurvey/imagine/blob/master/map/views.py. AstroDINO was trained with this stretch.
"""

import numpy as np

# band: (RGB plane, scale)
BAND_SCALES = {"g": (2, 6.0), "r": (1, 3.4), "z": (0, 2.2)}
SOFTENING = 0.03  # m
STRETCH = 20  # Q


def to_rgb(
    image: np.ndarray,
    bands: tuple[str, ...] = ("g", "r", "z"),
) -> np.ndarray:
    """Turn ``[H, W, 3]`` or ``[3, H, W]`` fluxes into a ``[H, W, 3]`` float32 RGB image.

    Applies an asinh stretch to the mean scaled intensity. Each band keeps its share of the intensity.
    """
    if image.shape[0] != len(bands):
        image = np.transpose(image, (2, 0, 1))
    intensity = 0
    for plane, band in zip(image, bands):
        intensity = intensity + np.maximum(0, plane * BAND_SCALES[band][1] + SOFTENING)
    intensity /= len(bands)
    stretched = np.arcsinh(STRETCH * intensity) / np.sqrt(STRETCH)
    intensity += (intensity == 0.0) * 1e-6
    rgb = np.zeros((*intensity.shape, 3), np.float32)
    for plane, band in zip(image, bands):
        channel, scale = BAND_SCALES[band]
        rgb[:, :, channel] = (plane * scale + SOFTENING) * stretched / intensity
    return np.clip(rgb, 0, 1)
