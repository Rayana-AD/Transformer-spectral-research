"""Publication-oriented Matplotlib settings (mathtext, no external LaTeX required)."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
