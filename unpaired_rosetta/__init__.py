"""Shared Geometry as a Rosetta Stone: cross-modal alignment without paired data."""

import os

# AVX-512 kernels round differently from AVX2 ones. Pinning MKL and PyTorch to AVX2 gives the same bits on every
# x86 machine. The variables only take effect before torch is imported. pixi sets them too.
os.environ.setdefault("MKL_CBWR", "AVX2,STRICT")
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")

from unpaired_rosetta.geometric_initialization import GeometricInitialization
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

__all__ = ["GeometricInitialization", "WassersteinProcrustes"]
