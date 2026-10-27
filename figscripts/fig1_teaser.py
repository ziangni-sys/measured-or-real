"""Figure 3: both running examples on one row, 7.0 x 2.0 in.

(a) Share of simulation runs in which the plug-in correction exceeds 1 in
    magnitude, and share in which Algorithm 1 declines to correct.
(b) Mean squared error of four corrections, log scale with a broken axis so
    the plug-in (about 1e7) and the bounded methods (below 1) both stay legible.
(c) ARC-Easy, Qwen2-1.5B minus Llama-3.2-3B: per-protocol simultaneous
    intervals of Algorithm 2 and the pooled comparison's interval.

Data: Gaussian rows of the robustness sweep (N=50, K=3, true correlation 0.4,
4,000 runs per cell), and the reported Algorithm 2 run (pairwise.csv,
pooled_pairwise.csv). Run from the repository root:
    python figscripts/fig1_teaser.py
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
from matplotlib.ticker import FixedLocator, NullLocator

import style

FIG_W, FIG_H = 7.0, 2.0
RHOS = [0.1, 0.2, 0.4, 0.6, 0.8]
PAIR = ("arc", "Qwen2-1.5B-Instruct", "Llama-3.2-3B-Instruct")


def check(cond, msg):
    if not cond:
        print("REPRODUCTION CHECK FAILED:", msg)
        sys.exit(1)
    print("check ok:", msg)


def fmt_gap(v):
    """Signed gap in points with a typographic minus."""
    return ("+" if v > 0 else "−") + f"{abs(v):.1f}"


# ---------------------------------------------------------------- data
rob = pd.read_csv(style.ROBUST)
gauss = rob[rob["noise"] == "gaussian"]
cfg = json.loads((style.ROBUST.parent / "config.json").read_text())
check((cfg["n"], cfg["k"], cfg["r_true"], cfg["alpha"], cfg["repetitions"]) == (50, 3, 0.4, 0.1, 4000),
      "sweep design: N=50, K=3, true correlation 0.4, alpha=0.1, 4,000 runs per cell")


def series(method, column):
    s = gauss[gauss["method"] == method].set_index("rho")[column].sort_index()
    if list(np.round(s.index, 3)) != RHOS:
        check(False, f"{method}: reliability grid is {RHOS}")
    return s


over_one = series("naive_plugin", "raw_over_one_rate")
declined = series("algorithm1", "refusal_or_gate_rate")
mse = {m: series(m, "mse") for m in ["naive_plugin", "clipped_plugin", "fixed_floor_0.1", "algorithm1"]}

pw = pd.read_csv(style.ALG2 / "pairwise.csv")
pp = pd.read_csv(style.ALG2 / "pooled_pairwise.csv")
key = (pw["benchmark"] == PAIR[0]) & (pw["model_a"] == PAIR[1]) & (pw["model_b"] == PAIR[2])
pkey = (pp["benchmark"] == PAIR[0]) & (pp["model_a"] == PAIR[1]) & (pp["model_b"] == PAIR[2])
check(key.sum() == 1 and pkey.sum() == 1, "one row for the ARC pair in each file")
row, prow = pw[key].iloc[0], pp[pkey].iloc[0]
rows_c = {  # gap, lower, upper in accuracy points
    "zero": 100 * row[["gap_protocol_0", "lower_protocol_0", "upper_protocol_0"]].to_numpy(float),
    "two": 100 * row[["gap_protocol_1", "lower_protocol_1", "upper_protocol_1"]].to_numpy(float),
    "pooled": 100 * prow[["gap", "lower", "upper"]].to_numpy(float),
}

# ---------------------------------------------------------------- reproduction checks
v = mse["naive_plugin"][0.1]
print(f"plug-in MSE at rho=0.1: {v:.4e}")
check(round(v / 1e6, 2) == 9.40, "plug-in MSE at rho=0.1 is 9.40e6")
for r, target in [(0.1, 0.152), (0.2, 0.124), (0.4, 0.075)]:
    v = mse["algorithm1"][r]
    print(f"Algorithm 1 MSE at rho={r}: {v:.5f}")
    check(abs(v - target) <= 0.0005, f"Algorithm 1 MSE at rho={r} is {target}")
v = over_one[0.1]
print(f"plug-in |r|>1 rate at rho=0.1: {v:.5f}")
check(abs(v - 0.633) <= 0.0005 and v > 0.60, "plug-in exceeds 1 in 0.633 (>0.60) of runs at rho=0.1")
v = declined[0.2]
print(f"Algorithm 1 declined rate at rho=0.2: {v:.5f}")
check(abs(v - 0.874) <= 0.0005, "Algorithm 1 declines in 0.874 of runs at rho=0.2")
z, t, p = rows_c["zero"], rows_c["two"], rows_c["pooled"]
print("zero-shot gap/lo/hi:", np.round(z, 3), " two-shot:", np.round(t, 3), " pooled:", np.round(p, 3))
check(abs(z[0] + 3.75) < 0.005 and abs(t[0] - 16.64) < 0.005, "gaps -3.75 / +16.64")
check(fmt_gap(z[0]) == "−3.8" and fmt_gap(t[0]) == "+16.6", "gaps print as -3.8 / +16.6")
check(np.allclose(np.round(z[1:], 1), [-5.8, -1.7]) and np.allclose(np.round(t[1:], 1), [14.6, 18.6]),
      "intervals [-5.8, -1.7] and [+14.6, +18.6]")
check(round(p[0], 1) == 6.4, "pooled gap +6.4")
check(row["verdict"] == "protocol-dependent" and prow["verdict"] == "stable-a",
      "verdicts: protocol-dependent (Algorithm 2), one winner (pooled)")
low = mse["naive_plugin"][mse["naive_plugin"] <= 1]
check(list(low.index) == [0.8] and np.isclose(low[0.8], mse["clipped_plugin"][0.8])
      and np.isclose(low[0.8], mse["fixed_floor_0.1"][0.8]),
      "plug-in MSE is below 1 only at 0.8, where it equals the clipped plug-in and the fixed floor")

# ---------------------------------------------------------------- layout (inches -> figure fractions)
fig = plt.figure(figsize=(FIG_W, FIG_H))


def axes_in(x0, y0, w, h):
    return fig.add_axes([x0 / FIG_W, y0 / FIG_H, w / FIG_W, h / FIG_H])


BOT, TOP = 0.30, 1.81           # axes area, inches from the bottom
H = TOP - BOT
XA, XB, XC = 0.34, 2.60, 4.97   # left edges of the three panels
WA, WB, WC = 1.70, 1.70, 1.99
TOP_H, GAP = 0.40, 0.07         # broken axis in (b): plug-in on top, bounded methods below
ax_a = axes_in(XA, BOT, WA, H)
ax_bt = axes_in(XB, TOP - TOP_H, WB, TOP_H)
ax_b = axes_in(XB, BOT, WB, H - TOP_H - GAP)
ax_c = axes_in(XC, BOT, WC, H)

XLIM = (0.06, 0.84)
for ax in (ax_a, ax_b, ax_bt):
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(FixedLocator(RHOS))
for ax in (ax_a, ax_b):
    ax.set_xticklabels([f"{r:g}" for r in RHOS])
    ax.set_xlabel("reliability of the 3-replicate mean", labelpad=2)


def line(ax, x, y, key, label=None, ms=3.5, clip=False, **over):
    enc = dict(style.METHOD[key])
    enc.update(over)
    ax.plot(x, y, color=enc["color"], marker=enc["marker"], ls=enc["ls"], lw=enc["lw"], ms=ms,
            label=label, zorder=enc.get("zorder", 3), clip_on=clip)


# ---------------------------------------------------------------- (a)
line(ax_a, over_one.index, over_one.values, "plugin")
line(ax_a, declined.index, declined.values, "ours", ls="--", lw=1.0)
ax_a.set_ylim(0, 1.0)
ax_a.yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
ax_a.set_yticklabels(["0", "", "0.5", "", "1"])
ax_a.set_ylabel("share of runs", labelpad=2)
ax_a.text(0.40, 0.60, "Algorithm 1\ndeclines", color=style.METHOD["ours"]["color"],
          fontsize=style.ANNOT, ha="left", va="center", linespacing=1.0)
ax_a.text(0.105, 0.135, "plug-in\nexceeds 1\nin magnitude", color=style.METHOD["plugin"]["color"],
          fontsize=style.ANNOT, ha="left", va="center", linespacing=1.0)
ax_a.set_title("(a) Impossible values and declines", loc="left", pad=4)

# ---------------------------------------------------------------- (b), broken log axis
for ax in (ax_b, ax_bt):
    ax.set_yscale("log")
# Equal values (plug-in, clipped plug-in and fixed floor at rho=0.8; clipped plug-in and
# fixed floor nearly so at 0.6) are drawn largest marker first. The plug-in point at 0.8
# would still be hidden under the others, so it is an open ring around them.
MS = {"clipped": 4.6, "plugin": 3.9, "floor": 3.4, "ours": 3.5}
for name, key in [("clipped_plugin", "clipped"), ("naive_plugin", "plugin"),
                  ("fixed_floor_0.1", "floor"), ("algorithm1", "ours")]:
    x, y = mse[name].index.to_numpy(), mse[name].to_numpy()
    for ax in (ax_b, ax_bt):
        if key == "plugin":
            # No segment crosses the break, which hides four decades.
            hi = y > 1
            line(ax, x[hi], y[hi], key, ms=MS[key], clip=True)
            ax.plot(x[~hi], y[~hi], ls="none", marker="o", ms=6.2, mfc="none",
                    mec=style.METHOD["plugin"]["color"], mew=0.9, zorder=6, clip_on=True)
        else:
            lab = style.METHOD[key]["label"] if ax is ax_b else None
            line(ax, x, y, key, label=lab, ms=MS[key], clip=True, zorder=4 if key == "ours" else 3)
ax_b.set_ylim(0.015, 1.5)
ax_b.yaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5, 1.0]))
ax_b.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1"])
ax_b.yaxis.set_minor_locator(NullLocator())
ax_bt.set_ylim(2e4, 3e7)
ax_bt.yaxis.set_major_locator(FixedLocator([1e5, 1e7]))
ax_bt.set_yticklabels([r"$10^5$", r"$10^7$"])
ax_bt.yaxis.set_minor_locator(NullLocator())
ax_bt.spines["bottom"].set_visible(False)
ax_bt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
# break marks on the left spine
d = 0.05  # inches
for y_sp in (TOP - TOP_H, BOT + H - TOP_H - GAP):
    fig.add_artist(plt.Line2D([XB - d, XB + d], [y_sp - d / 2, y_sp + d / 2], color="black", lw=0.6,
                              transform=fig.dpi_scale_trans, clip_on=False))
ax_bt.text(0.83, 4.5e6, r"plug-in (floor $10^{-9}$)", color=style.METHOD["plugin"]["color"],
           fontsize=style.ANNOT, ha="right", va="center")
ax_b.legend(loc="upper right", handlelength=1.6, borderaxespad=0.0, labelspacing=0.25, handletextpad=0.5)
fig.text((XB - 0.36) / FIG_W, (BOT + H / 2) / FIG_H, "mean squared error", rotation=90,
         ha="center", va="center", fontsize=8)
ax_bt.set_title("(b) Error of each correction", loc="left", pad=4)

# ---------------------------------------------------------------- (c)
ypos = {"zero": 2, "two": 1, "pooled": 0}
for k, (g, lo, hi) in rows_c.items():
    enc = style.COND[k]
    y = ypos[k]
    ax_c.plot([lo, hi], [y, y], color=enc["color"], lw=1.6, solid_capstyle="butt", zorder=2)
    ax_c.plot([g], [y], marker=enc["marker"], ms=5, color=enc["color"], mec=enc["color"],
              mfc=enc.get("mfc", enc["color"]), mew=1.0, ls="none", zorder=3)
    ax_c.text(g, y + 0.22, fmt_gap(g), ha="center", va="bottom", fontsize=style.ANNOT, color="black")
ax_c.axvline(0, color="0.55", lw=0.6, zorder=1)
ax_c.set_xlim(-8, 21)
ax_c.set_ylim(-0.55, 2.75)
ax_c.xaxis.set_major_locator(FixedLocator([-5, 0, 5, 10, 15, 20]))
ax_c.set_xticklabels(["−5", "0", "5", "10", "15", "20"])
ax_c.set_yticks([2, 1, 0])
ax_c.set_yticklabels(["zero-shot", "two-shot", "pooled"])
ax_c.tick_params(axis="y", length=0, pad=3)
ax_c.spines["left"].set_visible(False)
ax_c.set_xlabel("Qwen2-1.5B minus Llama-3.2-3B (points)", labelpad=2)
ax_c.set_title("(c) A reversal on ARC-Easy", loc="left", pad=4)

out = style.save(fig, "fig1_teaser")
page = pymupdf.open(out)[0].rect
print(f"final size: {page.width / 72:.3f} x {page.height / 72:.3f} in")
check(abs(page.width / 72 - FIG_W) < 0.03 and abs(page.height / 72 - FIG_H) < 0.03,
      f"saved size matches {FIG_W} x {FIG_H} in")
