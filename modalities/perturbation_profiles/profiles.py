"""Consensus profiles of the (compound, dose) units measured reproducibly by both assays.

1. Every treated well (Cell Painting) or signature (L1000) is z-scored per feature over all treated replicates of its
   assay. Doses are matched on log10 micromolar rounded to 0.1, since the two assays record the same nominal dose to
   different precision.
2. A unit is reproducible in an assay when the median pairwise Pearson correlation of its replicates exceeds the 95th
   percentile of a null of 2,000 random replicate groups of the same sizes ("percent replicating"). Most compounds have
   no effect, and noise cannot be aligned to noise, so only the units reproducible in both assays are kept (2,732).
3. The consensus profile of a unit is the median over its replicates.

Writes ``<assay>.npy`` (``[units, features]`` float32, row ``i`` the same unit in both) and ``index.json``
(compound id, name and log10 dose of every row).

    pixi run -e modalities python -m modalities.perturbation_profiles.profiles
"""

import json

import numpy as np

from modalities.perturbation_profiles import PROFILES, REPLICATES, profile_path

DOSE_DECIMALS = 1
MIN_REPLICATES = 2
NULL_DRAWS = 2000
NULL_PERCENTILE = 95


def load_replicates(
    assay: str,
) -> tuple[np.ndarray, list[tuple[str, float]], dict[tuple[str, float], str]]:
    """Z-scored features ``[replicates, d]``, the unit ``(compound id, log10 dose)`` of every row, and the name of
    every unit.
    """
    import pandas as pd

    frame = pd.read_parquet(REPLICATES[assay])
    metadata_columns = [column for column in frame.columns if column.startswith("Metadata_")]
    feature_columns = [column for column in frame.columns if column not in metadata_columns]
    frame = frame[frame["Metadata_pert_type"] == "trt"].copy()
    frame = frame[frame["Metadata_pert_dose_micromolar"].astype(float) > 0]
    frame["unit_dose"] = np.round(np.log10(frame["Metadata_pert_dose_micromolar"].astype(float)), DOSE_DECIMALS)
    features = np.nan_to_num(frame[feature_columns].to_numpy(dtype=np.float32))
    features -= features.mean(0, keepdims=True)
    features /= features.std(0, keepdims=True) + 1e-8
    units = list(zip(frame["Metadata_pert_id"], frame["unit_dose"]))
    return features, units, dict(zip(units, frame["Metadata_pert_iname"]))


def median_self_correlation(
    block: np.ndarray,
) -> float:
    """Median pairwise Pearson correlation between the rows of one unit's replicates."""
    centred = block - block.mean(1, keepdims=True)
    normed = centred / (np.linalg.norm(centred, axis=1, keepdims=True) + 1e-8)
    similarity = normed @ normed.T
    return float(np.median(similarity[np.triu_indices(len(block), k=1)]))


def reproducibility(
    features: np.ndarray,
    groups: dict[tuple[str, float], np.ndarray],
    generator: np.random.Generator,
) -> tuple[dict[tuple[str, float], float], float]:
    """Replicate correlation of every unit, and the 95th percentile of a null of random groups of the same sizes."""
    scores = {unit: median_self_correlation(features[rows]) for unit, rows in groups.items()}
    sizes = [len(rows) for rows in groups.values()]
    null = []
    for _ in range(NULL_DRAWS):
        size = sizes[generator.integers(len(sizes))]
        rows = generator.choice(len(features), size=size, replace=False)
        null.append(median_self_correlation(features[rows]))
    return scores, float(np.percentile(null, NULL_PERCENTILE))


def consensus_profiles() -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """The consensus profile of every unit reproducible in both assays, per assay, and the units in row order."""
    generator = np.random.default_rng(0)  # one stream for both assays, morphology first
    replicates, groups, scores, thresholds, names = {}, {}, {}, {}, {}
    for assay in REPLICATES:
        features, units, names[assay] = load_replicates(assay)
        rows_of_unit = {}
        for row, unit in enumerate(units):
            rows_of_unit.setdefault(unit, []).append(row)
        groups[assay] = {unit: np.array(rows) for unit, rows in rows_of_unit.items() if len(rows) >= MIN_REPLICATES}
        scores[assay], thresholds[assay] = reproducibility(features, groups[assay], generator)
        replicates[assay] = features
    shared = sorted(set(groups["morphology"]) & set(groups["expression"]))
    kept = [unit for unit in shared if all(scores[assay][unit] > thresholds[assay] for assay in REPLICATES)]
    profiles = {
        assay: np.stack([np.median(replicates[assay][groups[assay][unit]], axis=0) for unit in kept]).astype(np.float32)
        for assay in REPLICATES
    }
    index = [
        {"pert_id": unit[0], "pert_iname": str(names["morphology"].get(unit, "")), "dose": float(unit[1])}
        for unit in kept
    ]
    print(f"{len(shared)} units measured in both assays, {len(kept)} reproducible in both")
    return profiles, index


if __name__ == "__main__":
    profiles, index = consensus_profiles()
    PROFILES.mkdir(parents=True, exist_ok=True)
    for assay, profile in profiles.items():
        np.save(profile_path(assay), profile)
    (PROFILES / "index.json").write_text(json.dumps(index))
    print(f"wrote {PROFILES}: " + ", ".join(f"{assay} {profile.shape}" for assay, profile in profiles.items()))
