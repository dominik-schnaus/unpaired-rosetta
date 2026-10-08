"""Settings that depend on your machine. Every one comes from an environment variable and has a default, so the
code works without any of them.

``UNPAIRED_ROSETTA_ROOT``
    Folder for embeddings, data, results and figures. Default ``./storage``.
``UNPAIRED_ROSETTA_RAW``
    Folder for raw downloads (images, captions, benchmark files). Default ``<root>/raw``.
``UNPAIRED_ROSETTA_RESULTS`` and ``UNPAIRED_ROSETTA_FIGURES``
    Names of the results and figures folders below the root. Defaults ``results`` and ``figures``.
``ITSAMATCH_ROOT``
    Output folder of itsamatch, whose CIFAR-100 embeddings and class subsets the QAP benchmark (Figure 7) reuses.
    Default ``<root>/itsamatch``.
``UNPAIRED_ROSETTA_SLURM_PARTITION``
    SLURM partition of the jobs that ``--slurm`` submits. Default: the cluster's default partition.
``UNPAIRED_ROSETTA_SLURM_MAX_CPUS``
    Upper limit on the CPU cores per SLURM job. Default: no limit.
``UNPAIRED_ROSETTA_SLURM_EXCLUDE``
    Comma-separated SLURM nodes to avoid, e.g. nodes whose GPUs the CUDA builds do not support. Default: none.
"""

import os
from pathlib import Path


def storage_root() -> Path:
    return Path(os.environ.get("UNPAIRED_ROSETTA_ROOT", "storage")).expanduser()


def raw_root() -> Path:
    return Path(os.environ.get("UNPAIRED_ROSETTA_RAW", storage_root() / "raw")).expanduser()


def itsamatch_root() -> Path:
    return Path(os.environ.get("ITSAMATCH_ROOT", storage_root() / "itsamatch")).expanduser()


def results_folder_name() -> str:
    return os.environ.get("UNPAIRED_ROSETTA_RESULTS", "results")


def figures_folder_name() -> str:
    return os.environ.get("UNPAIRED_ROSETTA_FIGURES", "figures")


def slurm_partition() -> str | None:
    return os.environ.get("UNPAIRED_ROSETTA_SLURM_PARTITION") or None


def slurm_max_cpus() -> int | None:
    value = os.environ.get("UNPAIRED_ROSETTA_SLURM_MAX_CPUS")
    return int(value) if value else None


def slurm_exclude() -> str:
    return os.environ.get("UNPAIRED_ROSETTA_SLURM_EXCLUDE", "")
