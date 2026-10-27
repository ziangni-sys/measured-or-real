"""Figure D.1: Algorithm 1 under non-Gaussian noise, plus the Gaussian ablation.

Four panels at 7.0 x 2.1 in, log y. Panels (a)-(c): Gaussian, Student-t3 and
contaminated noise, with Algorithm 1, the bootstrap version of Algorithm 1,
the clipped plug-in and the fixed floor 0.1. Panel (d): Gaussian noise with
the components of Algorithm 1 removed one at a time (clipping only, upper
bound without the gate, gate with the point ICC) next to the full rule.

Data: the robustness sweep results.csv (columns noise, rho, method, mse).
Run from the repository root:
    python figscripts/figD1_robustness.py
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

import style

NOISE_PANELS = [("gaussian", "(a) Gaussian"),
                ("student_t3", r"(b) Student-$t_3$"),
                ("contaminated", "(c) Contaminated")]
NOISE_METHODS = ["algorithm1", "bootstrap_ucb", "clipped_plugin", "fixed_floor_0.1"]
ABLATION_METHODS = ["clipping_only", "ucb_no_gate", "gate_point", "algorithm1"]
YTICKS = [0.02, 0.05, 0.1, 0.2, 0.5]


def check(ok, msg):
    if not ok:
        sys.exit(f"REPRODUCTION CHECK FAILED: {msg}")


def run_checks(mse):
    """Table D.4 values, the rho=0.4 values, and the largest bootstrap gain."""
    g = mse.loc["gaussian"]
    table = {"clipping_only": (0.733, 0.176), "ucb_no_gate": (0.351, 0.083),
             "gate_point": (0.155, 0.096), "algorithm1": (0.152, 0.075)}
    for m, (v1, v4) in table.items():
        a, b = g.loc[0.1, m], g.loc[0.4, m]
        print(f"Gaussian {m:14s} rho=0.1 {a:.4f}  rho=0.4 {b:.4f}")
        check(round(a, 3) == v1 and round(b, 3) == v4, f"Table D.4 row {m}")
    at4 = [mse.loc[(n, 0.4), "algorithm1"] for n, _ in NOISE_PANELS]
    print("Algorithm 1 at rho=0.4 (Gaussian, t3, contaminated):", " / ".join(f"{v:.4f}" for v in at4))
    check([round(v, 3) for v in at4] == [0.075, 0.071, 0.071], "Algorithm 1 at rho=0.4")
    gain = (mse["algorithm1"] - mse["bootstrap_ucb"]) / mse["algorithm1"]
    print(f"largest bootstrap gain {100 * gain.max():.2f}% at {gain.idxmax()}")
    check(round(100 * gain.max(), 1) == 1.6, "largest bootstrap gain != 1.6%")
    # Clipping only and the clipped plug-in are the same estimator; panel (d) relies on that.
    check(np.allclose(mse["clipping_only"], mse["clipped_plugin"]), "clipping_only != clipped_plugin")


def line_kw(m):
    """Plot keywords for one CSV method, from the shared encodings in style.METHOD."""
    enc = style.METHOD[style.CSV_METHOD[m]]
    kw = dict(color=enc["color"], marker=enc["marker"], ls=enc["ls"], lw=enc["lw"],
              ms=3.6, zorder=3)
    if m == "bootstrap_ucb":
        # Drawn as an open ring on top of Algorithm 1, which it nearly coincides with.
        kw.update(mfc="none", mew=0.8, ms=5.2, zorder=4)
    if m == "algorithm1":
        kw.update(zorder=3.5)
    if m == "gate_point":
        kw.update(ms=4.0, mew=0.9)
    return kw


def draw(ax, sub, methods):
    for m in methods:
        ax.plot(sub[m].index, sub[m].values, **line_kw(m))


def main():
    d = pd.read_csv(style.ROBUST)
    mse = d.pivot_table(index=["noise", "rho"], columns="method", values="mse")
    run_checks(mse)

    fig, axes = plt.subplots(1, 4, figsize=(style.TEXT_W, 2.1), sharey=True,
                             layout="constrained")
    # Small outer pads, so the tight-cropped PDF keeps the specified width.
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)
    for ax, (noise, title) in zip(axes[:3], NOISE_PANELS):
        draw(ax, mse.loc[noise], NOISE_METHODS)
        ax.set_title(title, loc="left")
    draw(axes[3], mse.loc["gaussian"], ABLATION_METHODS)
    axes[3].set_title("(d) Gaussian, ablation", loc="left")

    rhos = sorted(d["rho"].unique())
    for ax in axes:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator(YTICKS))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels([f"{t:g}" for t in YTICKS])
        ax.set_xticks(rhos)
        ax.set_xticklabels([f"{r:g}" for r in rhos])
        ax.set_xlim(0.05, 0.85)
        ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    axes[0].set_ylim(0.018, 0.85)
    axes[0].set_ylabel("mean squared error")
    fig.supxlabel(r"reliability $\rho_K$ of the 3-replicate mean", fontsize=8)

    # One legend for the figure; "clipped plug-in" in panel (d) is the clipping-only ablation.
    order = ["algorithm1", "bootstrap_ucb", "clipped_plugin", "fixed_floor_0.1",
             "ucb_no_gate", "gate_point"]
    handles = [Line2D([], [], **line_kw(m)) for m in order]
    labels = [style.METHOD[style.CSV_METHOD[m]]["label"] for m in order]
    labels[2] = "clipped plug-in (clipping only)"
    fig.legend(handles, labels, loc="outside upper center", ncol=6, handlelength=2.4,
               columnspacing=1.4, handletextpad=0.5, borderaxespad=0.1)
    style.save(fig, "figD1_robustness")


if __name__ == "__main__":
    main()
