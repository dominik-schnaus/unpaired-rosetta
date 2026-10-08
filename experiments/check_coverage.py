"""Check that every figure and table of the paper has an experiment script and a generated PNG.

    python -m experiments.check_coverage [--figures figures]

For each item, it checks that the docstring of the script that makes it names the item, and that a PNG with the
item's prefix exists in ``$UNPAIRED_ROSETTA_ROOT/<figures>/<script>/``.
"""

import argparse
import importlib
import re

from unpaired_rosetta.embeddings import storage_root

# item of the paper -> (experiment script, prefix of the PNG it writes)
ITEMS = {
    "Fig. 1": ("vision_language_examples", "fig1_"),
    "Fig. 3a": ("unpaired", "fig3a_"),
    "Fig. 3b": ("unpaired", "fig3b_"),
    "Fig. 3c": ("shared_geometry", "fig3c_"),
    "Fig. 4a": ("domain_specific", "fig4a_"),
    "Fig. 4b": ("other_domains", "fig4b_"),
    "Fig. 5": ("few_pair", "fig5_"),
    "Fig. 6a": ("vision_language_examples", "fig6abc_"),
    "Fig. 6b": ("vision_language_examples", "fig6abc_"),
    "Fig. 6c": ("vision_language_examples", "fig6abc_"),
    "Fig. 6d": ("text_to_image", "fig6d_"),
    "Fig. 7": ("qap_solvers", "fig7"),
    "Fig. 8": ("ablation_hyperparameters", "fig8"),
    "Fig. 9": ("unpaired", "fig3a_fig9"),
    "Fig. 10": ("unpaired", "fig10_"),
    "Fig. 11": ("unpaired", "fig11_"),
    "Fig. 12": ("unpaired", "fig12_"),
    "Fig. 13": ("cross_dataset", "fig13_"),
    "Fig. 14": ("shared_geometry", "fig3c_fig14"),
    "Fig. 15": ("text_to_image", "fig15"),
    "Fig. 16": ("text_to_image", "fig16"),
    "Fig. 17": ("text_to_image", "fig17"),
    "Fig. 18": ("text_to_image", "fig18"),
    "Fig. 19": ("qualitative", "fig19"),
    "Fig. 20": ("token_pooling", "fig20"),
    "Fig. 21": ("token_pooling", "fig21"),
    "Fig. 22": ("granularity", "fig22"),
    "Fig. 23": ("granularity", "fig23"),
    "Fig. 24": ("granularity", "fig24"),
    "Tab. 1": ("ablation_initialization", "tab1"),
    "Tab. 2": ("lowrank_gw", "tab2"),
    "Tab. 3": ("ablation_readout", "tab3"),
    "Tab. 4": ("ablation_refinement", "tab4"),
    "Tab. 5": ("other_domains", "tab5"),
    "Tab. 6": ("text_to_image_scores", "tab6"),
    "Tab. 7": ("text_to_image_scores", "tab7"),
    "Tab. 8": ("domain_specific", "tab8"),
    "Tab. 9": ("domain_specific", "tab9"),
    "Tab. 10": ("domain_specific", "tab10"),
    "Tab. 11": ("cyclegan_pairs", "tab11"),
    "Tab. 12": ("other_domains", "tab12"),
}


def mentioned_numbers(
    docstring: str,
    kind: str,
) -> set[int]:
    """The figure or table numbers a docstring names. Ranges such as "Figs. 9-12" are expanded."""
    words = {"Fig.": r"(?:Figs?\.|Figures?)", "Tab.": r"(?:Tabs?\.|Tables?)"}[kind]
    numbers = set()
    for listing in re.findall(words + r"\s*((?:\d+[a-d]?(?:\s*[-–]\s*\d+)?[a-d]?(?:\s*(?:,|and|/)\s*)?)+)", docstring):
        for first, last in re.findall(r"(\d+)[a-d]?(?:\s*[-–]\s*(\d+))?", listing):
            numbers.update(range(int(first), int(last or first) + 1))
    return numbers


def mentions(
    docstring: str,
    item: str,
) -> bool:
    kind, number = item.split(" ")
    return int(number.rstrip("abcd")) in mentioned_numbers(docstring, kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    missing = 0
    for item, (script, prefix) in ITEMS.items():
        try:
            docstring = importlib.import_module(f"experiments.{script}").__doc__ or ""
            documented = mentions(docstring, item)
        except ModuleNotFoundError:
            documented = None
        pngs = sorted((storage_root() / args.figures / script).glob(f"{prefix}*.png"))
        status = "ok" if documented and pngs else "MISSING"
        missing += status != "ok"
        print(
            f"{item:8} {script:26} script {'-' if documented is None else ('ok' if documented else 'no mention'):10} "
            f"png {pngs[0].name if pngs else '-'}  {status}"
        )
    print(f"{len(ITEMS) - missing} of {len(ITEMS)} items covered")


if __name__ == "__main__":
    main()
