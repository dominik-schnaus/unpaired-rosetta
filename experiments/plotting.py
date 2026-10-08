"""Plotting helpers for the experiment scripts. Each figure or table is saved as a PNG with a CSV of its numbers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

OURS_COLOR = "#0065BD"  # our method is blue in every figure
PALETTE = ["#0065BD", "#E37222", "#A2AD00", "#98C6EA", "#DAD7CB", "#999999", "#64A0C8", "#C4071B", "#69085A", "#000000"]
FOSCTTM_COLORS = LinearSegmentedColormap.from_list("foscttm", ["#003359", "#0065BD", "#98C6EA", "#FFFFFF"])
ACCURACY_COLORS = LinearSegmentedColormap.from_list("accuracy", ["#FFFFFF", "#CADB8A", "#A2AD00", "#4E5200"])

sns.set_theme(context="paper", style="whitegrid", font_scale=0.9)


def save(
    figure: Figure,
    path: Path,
    data: pd.DataFrame | None = None,
) -> None:
    """Write ``<path>.png`` and, if given, the plotted numbers as ``<path>.csv``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)
    if data is not None:
        data.to_csv(path.with_suffix(".csv"), index=False)
    print(f"wrote {path.with_suffix('.png')}")


def mean_and_std(
    frame: pd.DataFrame,
    by: list[str],
    metric: str,
    ddof: int = 1,
) -> pd.DataFrame:
    """Mean and standard deviation of ``metric`` over the seeds of every group."""
    grouped = frame.groupby(by, sort=False)[metric]
    summary = grouped.agg(mean="mean", count="count").reset_index()
    summary["std"] = grouped.std(ddof=ddof).to_numpy()
    return summary


def heatmap(
    axis: Axes,
    means: pd.DataFrame,
    stds: pd.DataFrame | None,
    colors: Colormap,
    vmin: float,
    vmax: float,
    percent: bool = False,
) -> None:
    """A heatmap whose cells print the mean and, below it, the standard deviation."""
    sns.heatmap(means, ax=axis, cmap=colors, vmin=vmin, vmax=vmax, cbar=False, linewidths=0.5, linecolor="white")
    for row in range(means.shape[0]):
        for column in range(means.shape[1]):
            mean = means.iat[row, column]
            if np.isnan(mean):
                continue
            text = f"{100 * mean:.0f}" if percent else f"{mean:.2f}"
            if stds is not None and not np.isnan(stds.iat[row, column]):
                spread = f"{100 * stds.iat[row, column]:.0f}" if percent else f"{stds.iat[row, column]:.2f}"
                text += f"\n±{spread}"
            fraction = (mean - vmin) / (vmax - vmin)
            dark_cell = fraction < 0.45 if not percent else fraction > 0.6
            axis.text(
                column + 0.5,
                row + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=6,
                color="white" if dark_cell else "black",
            )
    axis.set_xlabel("")
    axis.set_ylabel("")


def linear_fit(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    clusters: pd.Series | np.ndarray | None = None,
    num_bootstrap: int = 2000,
    seed: int = 0,
    grid: int = 64,
) -> dict:
    """Least-squares line with R^2, Pearson's r, Spearman's rho and a 95 % band from a cluster bootstrap.

    The points are not independent because a model pair appears once per dataset. The bootstrap therefore resamples
    whole clusters, given by ``clusters``, instead of single points.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    clusters = np.arange(len(x)) if clusters is None else np.asarray(clusters)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    positions = np.linspace(x.min(), x.max(), grid)
    groups = [np.flatnonzero(clusters == name) for name in pd.unique(clusters)]
    generator = np.random.default_rng(seed)
    lines = []
    for _ in range(num_bootstrap):
        drawn = np.concatenate([groups[index] for index in generator.integers(0, len(groups), len(groups))])
        if len(np.unique(x[drawn])) < 2:
            continue
        bootstrap_slope, bootstrap_intercept = np.polyfit(x[drawn], y[drawn], 1)
        lines.append(bootstrap_slope * positions + bootstrap_intercept)
    band = np.quantile(np.array(lines), [0.025, 0.975], axis=0)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - residual.var() / y.var()),
        "pearson": float(stats.pearsonr(x, y).statistic),
        "spearman": float(stats.spearmanr(x, y).statistic),
        "num_points": len(x),
        "x": positions,
        "fit": slope * positions + intercept,
        "low": band[0],
        "high": band[1],
    }


def fit_scatter(
    axis: Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    category: str,
    fit: dict,
    baseline: float | None = 0.5,
) -> None:
    """Points coloured by ``category``, the fitted line with its band, and R^2 and Spearman's rho in the corner."""
    axis.fill_between(fit["x"], fit["low"], fit["high"], color="gray", alpha=0.15, linewidth=0)
    axis.plot(fit["x"], fit["fit"], color="#333333", linewidth=1)
    if baseline is not None:
        axis.axhline(baseline, color="gray", linestyle="--", linewidth=0.8)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    for index, (name, rows) in enumerate(frame.groupby(category, observed=True, sort=False)):
        axis.scatter(
            rows[x],
            rows[y],
            s=14,
            marker=markers[index % len(markers)],
            color=PALETTE[index % len(PALETTE)],
            edgecolor="#333333",
            linewidth=0.3,
            label=str(name),
            zorder=3,
        )
    axis.text(
        0.97,
        0.97,
        f"$R^2$ = {fit['r2']:.2f}\n$\\rho_s$ = {fit['spearman']:.2f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )


def table_figure(
    table: pd.DataFrame,
    title: str | None = None,
    bold: pd.DataFrame | None = None,
) -> Figure:
    """A table drawn as a figure. ``bold`` marks the cells to print in bold."""
    figure, axis = plt.subplots(figsize=(1.6 + 1.5 * table.shape[1], 0.4 + 0.3 * table.shape[0]))
    axis.axis("off")
    rendered = axis.table(cellText=table.to_numpy(), colLabels=list(table.columns), loc="center", cellLoc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8)
    rendered.scale(1, 1.3)
    for (row, column), cell in rendered.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_text_props(weight="bold")
        elif bold is not None and bold.iat[row - 1, column]:
            cell.set_text_props(weight="bold")
    if title:
        axis.set_title(title, fontsize=9)
    return figure


def format_mean_std(
    mean: float,
    std: float,
    digits: int = 3,
) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"
