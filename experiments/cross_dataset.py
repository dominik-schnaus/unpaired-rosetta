"""Cross-dataset alignment (Sec. 4.1): MS COCO images against Stanford Paragraph Captioning captions.

The two training sets come from different corpora, so no image has its caption on the other side.
Produces the "ours (dagger)" column of Fig. 3a (drawn by ``experiments/unpaired.py``) and Fig. 13 (zero-shot accuracy).

    python -m experiments.cross_dataset run
    python -m experiments.cross_dataset plot
"""

import matplotlib.pyplot as plt

from experiments import plotting
from experiments.alignment import AlignmentRun, run_alignment
from experiments.runner import Experiment, figures_dir, main, results_dir
from experiments.unpaired import CLASSIFICATION, accuracy_panels, readable
from modalities import image_captions
from unpaired_rosetta.randomness import SEEDS
from unpaired_rosetta.results import load_results


def configurations() -> list[AlignmentRun]:
    return [
        AlignmentRun(
            "coco_train2014",
            vision,
            language,
            seed,
            "ours",
            dataset_y="StanfordParagraphCaptioning",
            classification=CLASSIFICATION,
        )
        for vision in image_captions.VISION_MODELS
        for language in image_captions.LANGUAGE_MODELS
        for seed in SEEDS
    ]


def run(
    config: AlignmentRun,
    num_workers: int,
    num_threads: int,
) -> dict:
    _, metrics = run_alignment(config, num_workers, num_threads)
    return metrics


def plot() -> None:
    frame = readable(load_results(results_dir("cross_dataset")))
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"wspace": 0.35})
    numbers = accuracy_panels(axes, frame)
    for axis in axes[1:]:
        axis.set_yticklabels([])
    figure.suptitle("Zero-shot accuracy, MS COCO images x SPC captions", fontsize=10)
    plotting.save(figure, figures_dir("cross_dataset") / "fig13_zero_shot", numbers)


EXPERIMENT = Experiment("cross_dataset", configurations, run, plot, memory_gb=96, hours=12)

if __name__ == "__main__":
    main(EXPERIMENT)
