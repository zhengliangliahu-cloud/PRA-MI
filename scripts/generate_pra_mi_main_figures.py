from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "writing" / "cmpb_pra_mi" / "figures"

COLORS = {
    "navy": "#17324D",
    "blue": "#2B6CB0",
    "teal": "#1B9A99",
    "amber": "#E69F00",
    "red": "#D55E00",
    "gray": "#6B7280",
    "light_gray": "#F4F6F8",
    "line": "#334155",
}


def setup_ax(width: float = 6.9, height: float = 3.8):
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def box(ax, xy, wh, title, body, fc="white", ec=COLORS["line"], title_color=None):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.15,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=title_color or ec,
    )
    ax.text(
        x + w / 2,
        y + h * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=7.4,
        color=COLORS["gray"],
        linespacing=1.15,
    )


def arrow(ax, start, end, color=COLORS["line"], rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def draw_wave(ax, x0, y0, w, amp=0.018, color=COLORS["teal"]):
    xs = np.linspace(x0, x0 + w, 90)
    ys = y0 + amp * np.sin(np.linspace(0, 6 * np.pi, xs.size))
    ax.plot(xs, ys, color=color, lw=1.4)


def draw_electrodes(ax, center, radius=0.055, color=COLORS["blue"]):
    cx, cy = center
    ax.add_patch(Circle((cx, cy), radius, fill=False, lw=1.0, ec=color))
    for dx, dy in [(0, 0), (-0.022, 0.018), (0.022, 0.018), (-0.022, -0.018), (0.022, -0.018)]:
        ax.add_patch(Circle((cx + dx, cy + dy), 0.006, color=color, alpha=0.9))


def save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def workflow():
    fig, ax = setup_ax()
    ax.text(0.03, 0.94, "PRA-MI residual-audit workflow", fontsize=12.5, fontweight="bold", color=COLORS["navy"])
    ax.text(0.03, 0.885, "Protocol first, then performance and residual interpretation", fontsize=8.2, color=COLORS["gray"])

    # Lane backgrounds.
    ax.add_patch(FancyBboxPatch((0.025, 0.54), 0.95, 0.27, boxstyle="round,pad=0.01,rounding_size=0.018", fc="#F7FAFC", ec="#D8DEE9", lw=0.8))
    ax.add_patch(FancyBboxPatch((0.025, 0.18), 0.95, 0.27, boxstyle="round,pad=0.01,rounding_size=0.018", fc="#FFF8ED", ec="#F0D49A", lw=0.8))
    ax.text(0.045, 0.785, "Evaluation lane", fontsize=8.5, fontweight="bold", color=COLORS["blue"])
    ax.text(0.045, 0.425, "Diagnostic lane", fontsize=8.5, fontweight="bold", color=COLORS["amber"])

    top_y, bot_y = 0.60, 0.24
    w, h = 0.205, 0.15
    xs = [0.09, 0.385, 0.68]
    box(ax, (xs[0], top_y), (w, h), "Public MI EEG", "BNCI2014_001\nCho2017 / Lee2019", fc="white", ec=COLORS["blue"])
    draw_electrodes(ax, (xs[0] + 0.028, top_y + 0.114), radius=0.026, color=COLORS["blue"])
    box(ax, (xs[1], top_y), (w, h), "LOSO split", "source train/val\nheld-out target", fc="white", ec=COLORS["blue"])
    box(ax, (xs[2], top_y), (w, h), "Audit record", "P0/P1 ledger\navailability trace", fc="white", ec=COLORS["blue"])
    arrow(ax, (xs[0] + w, top_y + h / 2), (xs[1], top_y + h / 2), COLORS["blue"])
    arrow(ax, (xs[1] + w, top_y + h / 2), (xs[2], top_y + h / 2), COLORS["blue"])

    box(ax, (xs[0], bot_y), (w, h), "P0-safe QRI", "trial and channel\nreliability", fc="white", ec=COLORS["teal"])
    draw_wave(ax, xs[0] + 0.025, bot_y + 0.112, 0.065, color=COLORS["teal"])
    box(ax, (xs[1], bot_y), (w, h), "Decoders", "robust baselines\nQCA probe", fc="white", ec=COLORS["teal"])
    box(ax, (xs[2], bot_y), (w, h), "Evidence outputs", "balanced accuracy\nresidual failure", fc="white", ec=COLORS["teal"])
    arrow(ax, (xs[0] + w, bot_y + h / 2), (xs[1], bot_y + h / 2), COLORS["teal"])
    arrow(ax, (xs[1] + w, bot_y + h / 2), (xs[2], bot_y + h / 2), COLORS["teal"])

    arrow(ax, (xs[2] + w * 0.55, bot_y + h + 0.015), (xs[2] + w * 0.55, top_y - 0.015), COLORS["amber"])
    ax.text(xs[2] + w * 0.63, 0.505, "interpret\nunder protocol", fontsize=7.3, color=COLORS["gray"], va="center")
    save(fig, "pra_mi_workflow")


def protocol_availability():
    fig, ax = setup_ax()
    ax.text(0.03, 0.94, "Information availability boundaries", fontsize=12.5, fontweight="bold", color=COLORS["navy"])
    ax.text(0.03, 0.885, "PRA-MI separates model-side inputs from adaptation and post-hoc analysis", fontsize=8.2, color=COLORS["gray"])

    tiers = [
        (0.66, "P0", "Strict source-only / current-trial", "source data, current trial,\nD0a/D0b diagnostics", COLORS["teal"], "#ECFDF5"),
        (0.40, "P1", "Unlabeled target-stat adaptation", "P0 inputs plus target\naggregate statistics", COLORS["amber"], "#FFF7ED"),
        (0.14, "D3", "Post-hoc target-label / oracle", "target labels or accuracy\nproxies for analysis only", COLORS["red"], "#FFF1F2"),
    ]
    for y, code, title, body, color, fill in tiers:
        ax.add_patch(FancyBboxPatch((0.07, y), 0.86, 0.18, boxstyle="round,pad=0.018,rounding_size=0.025", fc=fill, ec=color, lw=1.15))
        ax.add_patch(Circle((0.13, y + 0.09), 0.045, fc=color, ec=color, alpha=0.95))
        ax.text(0.13, y + 0.09, code, ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax.text(0.21, y + 0.117, title, ha="left", va="center", fontsize=9.5, fontweight="bold", color=COLORS["navy"])
        ax.text(0.21, y + 0.062, body, ha="left", va="center", fontsize=7.8, color=COLORS["gray"], linespacing=1.15)

    # Icons and gates.
    draw_electrodes(ax, (0.73, 0.75), radius=0.035, color=COLORS["teal"])
    draw_wave(ax, 0.785, 0.75, 0.09, color=COLORS["teal"])
    ax.bar([0.73, 0.76, 0.79, 0.82], [0.035, 0.055, 0.045, 0.075], width=0.012, bottom=0.445, color=COLORS["amber"], alpha=0.9)
    ax.text(0.845, 0.49, "aggregate\nstatistics", fontsize=7.2, color=COLORS["gray"], va="center")
    ax.text(0.73, 0.225, "label", fontsize=7.0, color="white", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.22", fc=COLORS["red"], ec=COLORS["red"]))
    ax.text(0.805, 0.225, "oracle", fontsize=7.0, color="white", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.22", fc=COLORS["red"], ec=COLORS["red"]))

    for y, label in [(0.615, "target-stat boundary"), (0.355, "post-hoc boundary")]:
        ax.plot([0.09, 0.91], [y, y], color="#CBD5E1", lw=1.1, ls=(0, (4, 3)))
        ax.text(0.50, y + 0.014, label, ha="center", fontsize=7.2, color=COLORS["gray"], bbox=dict(fc="white", ec="none", pad=1.0))

    ax.text(0.07, 0.055, "Reported P0 runs must not use P1 aggregate statistics or D3 oracle information.", fontsize=7.7, color=COLORS["navy"])
    save(fig, "protocol_availability")


def main():
    workflow()
    protocol_availability()


if __name__ == "__main__":
    main()
