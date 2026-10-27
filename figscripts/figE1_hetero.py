"""Figure E.1: does a difficulty score's seed noise depend on its level?

Four hexbin panels at 7.0 x 1.8 in: for every training example, the SD of its
score over the 8 seeds against its mean over the 8 seeds, with a log colour
scale for the number of examples per hexagon. Panels: CIFAR-100 EL2N,
CIFAR-100 dynamic uncertainty, CIFAR-10 EL2N, CIFAR-10 dynamic uncertainty.
Each panel prints S, the Spearman correlation across examples between the
within-example variance (ddof=1) and the mean.

Data: data/prune_scores/seed{0..7}_e30.npz and data/prune_scores_cifar10/
seed{0..7}_e30.npz, keys el and du. Run from the repository root:
    python figscripts/figE1_hetero.py
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.stats import spearmanr

import matplotlib.patheffects as pe

import style

CELLS = [  # (folder, key, dataset, score, expected S at 3 decimals)
    ("prune_scores", "el", "CIFAR-100", "EL2N", 0.149),
    ("prune_scores", "du", "CIFAR-100", "Dynamic uncertainty", -0.382),
    ("prune_scores_cifar10", "el", "CIFAR-10", "EL2N", 0.737),
    ("prune_scores_cifar10", "du", "CIFAR-10", "Dynamic uncertainty", 0.100),
]
SEEDS = range(8)
CMAP = LinearSegmentedColormap.from_list("grey_seq", ["#D9D9D9", "#7A7A7A", "#1A1A1A"])


def check(ok, msg):
    if not ok:
        sys.exit(f"REPRODUCTION CHECK FAILED: {msg}")


def check_size(path, w, h, tol=0.04):
    import pymupdf
    with pymupdf.open(path) as doc:
        r = doc[0].rect
    ww, hh = r.width / 72, r.height / 72
    print(f"PDF size {ww:.3f} x {hh:.3f} in (target {w} x {h})")
    check(abs(ww - w) <= tol and abs(hh - h) <= tol, f"PDF size {ww:.2f} x {hh:.2f} in")


def load(folder, key):
    files = [style.REPO / "data" / folder / f"seed{s}_e30.npz" for s in SEEDS]
    X = np.stack([np.load(f)[key] for f in files]).T.astype(float)  # examples x seeds
    return X


def main():
    data = []
    for folder, key, ds, score, expect in CELLS:
        X = load(folder, key)
        check(X.shape == (50000, 8), f"{folder}/{key}: shape {X.shape}")
        mu, var = X.mean(1), X.var(1, ddof=1)
        S = spearmanr(mu, var).statistic
        print(f"{ds:9s} {score:20s} S = {S:+.3f}  (printed {S:+.2f})")
        check(round(S, 3) == expect, f"{ds} {score}: S = {S:+.4f}, expected {expect:+.3f}")
        data.append((mu, np.sqrt(var), S))

    fig = plt.figure(figsize=(style.TEXT_W, 1.8), layout="constrained")
    # Small outer pads keep every element on the canvas, which is saved at its exact size.
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)
    # One sub-figure per dataset, so each pair of panels gets a dataset header.
    # The right one is wider because it also holds the shared colour bar.
    sfigs = fig.subfigures(1, 2, width_ratios=[1.0, 1.1], wspace=0.0)
    axes = [ax for sf in sfigs for ax in sf.subplots(1, 2)]
    for sf, ds in zip(sfigs, ["CIFAR-100", "CIFAR-10"]):
        sf.suptitle(ds, fontsize=8)
    norm = LogNorm(vmin=1, vmax=10)  # vmax is reset to the largest hexagon count below
    for i, (ax, (mu, sd, S), (_, _, _, score, _)) in enumerate(zip(axes, data, CELLS)):
        xmax = mu.max() * 1.02  # every example is drawn
        ymax = sd.max() * 1.12  # headroom for the S label
        hb = ax.hexbin(mu, sd, gridsize=(34, 22), extent=(0, xmax, 0, ymax), cmap=CMAP,
                       norm=norm, mincnt=1, linewidths=0.15, edgecolors="face")
        # Median SD within 20 equal-count bins of the mean: where most examples sit.
        edges = np.quantile(mu, np.linspace(0, 1, 21))
        idx = np.clip(np.searchsorted(edges, mu, side="right") - 1, 0, 19)
        bx = np.array([np.median(mu[idx == b]) for b in range(20)])
        by = np.array([np.median(sd[idx == b]) for b in range(20)])
        ax.plot(bx, by, color="black", lw=0.9, marker="o", ms=1.8, mfc="white", mew=0.6,
                zorder=3, path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.set_title(f"({'abcd'[i]}) {score}", loc="left")
        # The upper-left corner is empty in every panel (no example has a mean near
        # zero and a large SD), so S goes there.
        ax.text(0.04, 0.98, f"$S={S:+.2f}$", transform=ax.transAxes, ha="left", va="top",
                fontsize=style.ANNOT)
        ax.set_xlabel("mean over 8 seeds")
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=4)
    counts = [c.get_array().max() for ax in axes for c in ax.collections]
    norm.vmax = max(counts)
    n_drawn = [int(c.get_array().sum()) for ax in axes for c in ax.collections]
    print("examples drawn per panel:", n_drawn, " largest hexagon count:", int(norm.vmax))
    check(all(n == 50000 for n in n_drawn), "some examples fall outside the hexbin extent")
    axes[0].set_ylabel("SD over 8 seeds")
    cb = sfigs[1].colorbar(hb, ax=axes[2:], fraction=0.05, pad=0.02, aspect=20)
    cb.set_ticks([1, 10, 100, 1000], labels=["1", "10", "100", "1000"])
    cb.ax.minorticks_off()
    cb.set_label("examples per hexagon", fontsize=7)
    cb.ax.tick_params(labelsize=7, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    # Centre each dataset header over its two panels. suptitle centres it on the whole
    # sub-figure, which also holds the y-label (left) or the colour bar (right).
    fig.canvas.draw()
    fig.set_layout_engine("none")  # freeze the constrained layout before moving the headers
    for sf, (a, b) in zip(sfigs, [(axes[0], axes[1]), (axes[2], axes[3])]):
        x0, x1 = a.get_window_extent().x0, b.get_window_extent().x1
        sf._suptitle.set_x(((x0 + x1) / 2 - sf.bbox.x0) / sf.bbox.width)
    out = style.save(fig, "figE1_hetero", tight=False)
    check_size(out, style.TEXT_W, 1.8)


if __name__ == "__main__":
    main()
