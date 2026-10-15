"""Shared chart style for every analysis in the portfolio."""

from pathlib import Path

import matplotlib.pyplot as plt

SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
# Categorical slots in fixed order (validated colour-blind-safe palette)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA = SERIES[:3]


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 110,
    })


def hbar(ax, labels, values, color=BLUE, fmt="{:,.0f}", highlight=None):
    """Horizontal bars, largest on top, values labelled directly."""
    labels, values = list(labels)[::-1], list(values)[::-1]
    colors = [color if highlight is None or l in highlight else MUTED for l in labels]
    ax.barh(labels, values, color=colors, height=0.65)
    xmax = max(values)
    for i, v in enumerate(values):
        ax.text(v + xmax * 0.01, i, fmt.format(v), va="center", fontsize=9.5, color=INK)
    ax.set_xlim(0, xmax * 1.15)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def save(fig, name: str, folder: str = "figures"):
    out = Path(folder)
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
