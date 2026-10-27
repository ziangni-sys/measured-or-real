"""Figure B.1: the synthetic persistence study.

Two panels at 7.0 x 2.2 in. Panel (a) shows the within-run estimator, panel (b)
the cross-replicate estimator with disattenuation. x is the true AR(1)
coefficient phi, y the estimate, with the central 95% of the 40 runs per
regime as a band. Line style encodes the readout noise level.

Data: data/synth.csv (columns alpha, sigma, truth, naive, naive_lo, naive_hi,
disatt, disatt_lo, disatt_hi). Run from the repository root:
    python figscripts/figB1_persistence.py
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import style

ESTIMATORS = [
    # (csv prefix, panel title, colour)
    ("naive", "(a) Within-run estimator", style.METHOD["plugin"]["color"]),
    ("disatt", "(b) Cross-replicate estimator with disattenuation", style.METHOD["ours"]["color"]),
]
NOISE_LS = {0.5: "-", 1.0: (0, (5, 1.5, 1, 1.5)), 2.0: (0, (1, 1.3))}


def check(ok, msg):
    if not ok:
        sys.exit(f"REPRODUCTION CHECK FAILED: {msg}")


def main():
    s = pd.read_csv(style.REPO / "data" / "synth.csv")
    s["sigma"] = s["sigma"].astype(float)

    # Reproduction checks: MAE 0.4404 vs. 0.0007, coverage 0/27 vs. 27/27.
    stats = {}
    for key, _, _ in ESTIMATORS:
        mae = (s[key] - s["truth"]).abs().mean()
        cov = int(((s[f"{key}_lo"] <= s["truth"]) & (s["truth"] <= s[f"{key}_hi"])).sum())
        stats[key] = (mae, cov)
        name = "within-run" if key == "naive" else "cross-replicate"
        print(f"{name:15s} MAE = {mae:.5f}   coverage = {cov}/{len(s)}")
    check(len(s) == 27, f"expected 27 regimes, found {len(s)}")
    check(round(stats["naive"][0], 4) == 0.4404, "within-run MAE != 0.4404")
    check(round(stats["disatt"][0], 4) == 0.0007, "cross-replicate MAE != 0.0007")
    check(stats["naive"][1] == 0, "within-run coverage != 0/27")
    check(stats["disatt"][1] == 27, "cross-replicate coverage != 27/27")
    check(sorted(s["sigma"].unique()) == sorted(NOISE_LS), "unexpected noise levels")

    fig, axes = plt.subplots(1, 2, figsize=(style.TEXT_W, 2.2), sharex=True, sharey=True,
                             layout="constrained")
    # Small outer pads, so the tight-cropped PDF keeps the specified width.
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)
    lims = (-0.5, 0.7)
    for ax, (key, title, col) in zip(axes, ESTIMATORS):
        ax.axhline(0, color="0.88", lw=0.5, zorder=0)
        ax.plot(lims, lims, color="0.55", lw=0.7, ls=(0, (4, 3)), zorder=1)
        for sg, ls in NOISE_LS.items():
            r = s[s["sigma"] == sg].sort_values("truth")
            ax.fill_between(r["truth"], r[f"{key}_lo"], r[f"{key}_hi"], color=col,
                            alpha=0.16, lw=0, zorder=2)
            ax.plot(r["truth"], r[key], color=col, ls=ls, lw=1.1, zorder=3)
        mae, cov = stats[key]
        ax.text(0.03, 0.97, f"mean absolute error {mae:.2g}\ntruth inside band: {cov}/{len(s)} regimes",
                transform=ax.transAxes, ha="left", va="top", fontsize=style.ANNOT,
                linespacing=1.3)
        ax.set_title(title, loc="left")
        ax.set_xlabel(r"true persistence $\phi$")
        ax.set_xticks(np.arange(-0.2, 0.61, 0.2))
        ax.set_yticks(np.arange(-0.4, 0.61, 0.2))
    axes[0].set_ylabel(r"estimate of $\phi$")
    axes[0].set_xlim(-0.24, 0.64)
    axes[0].set_ylim(*lims)
    # Direct label for the reference line, placed where no data are drawn.
    axes[0].text(0.52, 0.565, "$y=x$", color="0.45", fontsize=style.ANNOT, rotation=0,
                 ha="right", va="bottom")

    handles = [Line2D([], [], color="0.2", ls=ls, lw=1.1) for ls in NOISE_LS.values()]
    labels = [f"readout noise SD {sg:g}" for sg in NOISE_LS]
    axes[1].legend(handles, labels, loc="lower right", handlelength=2.6,
                   borderaxespad=0.2, labelspacing=0.3)
    style.save(fig, "figB1_persistence")


if __name__ == "__main__":
    main()
