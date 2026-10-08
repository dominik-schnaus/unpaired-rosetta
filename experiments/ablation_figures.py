"""Draws the ablation tables and figures (Appendix A.2) as PNG files, with a CSV and a Markdown copy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from experiments.runner import figures_dir

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# One color per series in a fixed order, never cycled.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
GRID = "#e4e3df"
DATASET_LABELS = {"nq": "NQ", "SNARE-seq": "SNARE-seq", "coco_train2014": "MS COCO"}


def set_style() -> None:
    sns.set_theme(style="whitegrid", rc={"grid.color": GRID, "axes.edgecolor": GRID, "text.color": TEXT})


def mean_std(
    values: pd.Series,
    digits: int = 3,
) -> str:
    """``mean +- std`` over the seeds, with the sample standard deviation as in the paper."""
    std = values.std() if len(values) > 1 else 0.0
    return f"{values.mean():.{digits}f} ± {std:.{digits}f}"


def save_table(
    table: pd.DataFrame,
    experiment: str,
    name: str,
    title: str,
) -> Path:
    """Write a table as CSV, Markdown and PNG. The index becomes the first column."""
    folder = figures_dir(experiment)
    table.to_csv(folder / f"{name}.csv")
    cells = table.reset_index()
    cells.columns = [
        " ".join(map(str, column)).strip() if isinstance(column, tuple) else str(column) for column in cells.columns
    ]
    cells = cells.map(lambda value: "--" if pd.isna(value) else str(value))
    (folder / f"{name}.md").write_text(f"**{title}**\n\n{markdown(cells)}\n")
    figure, axes = plt.subplots(figsize=(4, 0.3 * (len(cells) + 2)))
    axes.axis("off")
    rendered = axes.table(
        cellText=cells.values, colLabels=[str(column) for column in cells.columns], loc="center", cellLoc="center"
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.auto_set_column_width(list(range(len(cells.columns))))
    rendered.scale(1, 1.3)
    for (row, _), cell in rendered.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.get_text().set_color(TEXT)
        if row == 0:
            cell.get_text().set_fontweight("bold")
    axes.set_title(title, fontsize=10, color=TEXT)
    path = folder / f"{name}.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")
    return path


def markdown(
    cells: pd.DataFrame,
) -> str:
    lines = ["| " + " | ".join(map(str, cells.columns)) + " |", "|" + "---|" * len(cells.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.values.tolist()]
    return "\n".join(lines)


def save_figure(
    figure: Figure,
    experiment: str,
    name: str,
) -> Path:
    path = figures_dir(experiment) / f"{name}.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")
    return path
