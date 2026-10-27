"""Figure G.4: the 16 protocol-dependent pairs, 3.3 x 3.4 in.

One row per pair (model_a vs. model_b; the gap is model_a minus model_b, in
accuracy points). Per row: Algorithm 2's simultaneous intervals at zero-shot
(open square) and two-shot (filled square), and the pooled comparison's interval
for the average of the two settings (diamond). Rows are grouped by benchmark
and sorted by the zero-shot gap within each group.

Data: the reported Algorithm 2 run, pairwise.csv (verdict == protocol-dependent)
merged with pooled_pairwise.csv on (benchmark, model_a, model_b).
Run from the repository root:
    python figscripts/fig2_reversals.py
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import blended_transform_factory

import style

FIG_W, FIG_H = 3.3, 3.4
BENCH = [("arc", "ARC-Easy"), ("hellaswag", "HellaSwag"), ("mmlu", "MMLU"), ("openbookqa", "OpenBookQA")]
KEYS = ["benchmark", "model_a", "model_b"]


def check(cond, msg):
    if not cond:
        print("REPRODUCTION CHECK FAILED:", msg)
        sys.exit(1)
    print("check ok:", msg)


def short(m):
    return m.replace("-Instruct", "")


def fmt(v):
    """Signed value for the console log; ASCII so any console encoding can print it."""
    return ("+" if v > 0 else "-") + f"{abs(v):.1f}"


# ---------------------------------------------------------------- data
pw = pd.read_csv(style.ALG2 / "pairwise.csv")
pp = pd.read_csv(style.ALG2 / "pooled_pairwise.csv")
rev = pw[pw["verdict"] == "protocol-dependent"]
df = rev.merge(pp[KEYS + ["gap", "lower", "upper", "verdict"]], on=KEYS, how="left",
               suffixes=("", "_pooled"), validate="one_to_one")
for c in ["gap_protocol_0", "lower_protocol_0", "upper_protocol_0", "gap_protocol_1", "lower_protocol_1",
          "upper_protocol_1", "gap", "lower", "upper"]:
    df[c] = 100 * df[c]
order = {b: i for i, (b, _) in enumerate(BENCH)}
df = df.assign(_g=df["benchmark"].map(order)).sort_values(["_g", "gap_protocol_0"]).reset_index(drop=True)

# ---------------------------------------------------------------- reproduction checks
for _, r in df.iterrows():
    print(f"{r.benchmark:<11}{short(r.model_a):>13} vs. {short(r.model_b):<13}"
          f" zero {fmt(r.gap_protocol_0)} [{fmt(r.lower_protocol_0)}, {fmt(r.upper_protocol_0)}]"
          f"  two {fmt(r.gap_protocol_1)} [{fmt(r.lower_protocol_1)}, {fmt(r.upper_protocol_1)}]"
          f"  pooled {fmt(r.gap)} [{fmt(r.lower)}, {fmt(r.upper)}] {r.verdict_pooled}")
counts = df.groupby("benchmark").size().to_dict()
check(len(df) == 16, "16 protocol-dependent pairs")
check(counts == {"arc": 5, "hellaswag": 4, "mmlu": 3, "openbookqa": 4},
      "per benchmark: ARC 5, HellaSwag 4, MMLU 3, OpenBookQA 4")
check(df["gap"].notna().all(), "every pair has a pooled row")
named = df["verdict_pooled"].isin(["stable-a", "stable-b"])
check(named.sum() == 15, "pooled comparison names one winner for 15 of 16")
exc = df[~named]
check(len(exc) == 1 and exc.iloc[0][KEYS].tolist() == ["arc", "Qwen2.5-0.5B-Instruct", "Qwen2-0.5B-Instruct"],
      "the exception is Qwen2.5-0.5B vs. Qwen2-0.5B on ARC")
e = exc.iloc[0]
check(np.allclose(np.round([e.gap, e.lower, e.upper], 1), [-0.4, -2.3, 1.6]),
      "exception: pooled -0.4 [-2.3, +1.6]")
check(np.allclose(np.round([e.gap_protocol_0, e.lower_protocol_0, e.upper_protocol_0], 1), [6.1, 3.9, 8.4])
      and np.allclose(np.round([e.gap_protocol_1, e.lower_protocol_1, e.upper_protocol_1], 1), [-6.9, -9.2, -4.5]),
      "Qwen2.5-0.5B vs. Qwen2-0.5B: +6.1 [+3.9, +8.4] / -6.9 [-9.2, -4.5]")
# Every row is a reversal: both protocol intervals exclude zero, with opposite signs.
sign0 = np.sign(df["lower_protocol_0"]) * (np.sign(df["lower_protocol_0"]) == np.sign(df["upper_protocol_0"]))
sign1 = np.sign(df["lower_protocol_1"]) * (np.sign(df["lower_protocol_1"]) == np.sign(df["upper_protocol_1"]))
check(((sign0 * sign1) == -1).all(), "every row has opposite-signed intervals that exclude zero")

# ---------------------------------------------------------------- layout
ROW, HEAD = 1.0, 1.05       # vertical units per pair row and per benchmark header
OFF = 0.21                  # vertical offset of the zero-shot / pooled intervals within a row
ys, heads, banded, y = [], [], [], 0.0
for b, name in BENCH:
    y -= HEAD
    heads.append((y + 0.38, name))
    for i in range(int((df["benchmark"] == b).sum())):
        y -= ROW
        ys.append(y + ROW / 2)
        if i % 2 == 0:  # alternate light bands within each benchmark, starting with its first row
            banded.append(y + ROW / 2)
y_bottom = y

fig = plt.figure(figsize=(FIG_W, FIG_H))
X0, Y0, W, HT = 1.36, 0.33, 1.93, 2.83  # axes box in inches
ax = fig.add_axes([X0 / FIG_W, Y0 / FIG_H, W / FIG_W, HT / FIG_H])
ax.set_ylim(y_bottom - 0.15, 0.05)
ax.set_xlim(-22, 23)
ax.axvline(0, color="0.55", lw=0.6, zorder=1)
for yc in banded:
    ax.axhspan(yc - ROW / 2, yc + ROW / 2, color="0.95", lw=0, zorder=0)

parts = [("zero", "gap_protocol_0", "lower_protocol_0", "upper_protocol_0", +OFF),
         ("two", "gap_protocol_1", "lower_protocol_1", "upper_protocol_1", 0.0),
         ("pooled", "gap", "lower", "upper", -OFF)]
for yc, (_, r) in zip(ys, df.iterrows()):
    for k, g, lo, hi, dy in parts:
        enc = style.COND[k]
        ax.plot([r[lo], r[hi]], [yc + dy, yc + dy], color=enc["color"], lw=1.0, solid_capstyle="butt", zorder=2)
        ax.plot([r[g]], [yc + dy], ls="none", marker=enc["marker"], ms=3.4 if k != "pooled" else 3.8,
                color=enc["color"], mec=enc["color"], mfc=enc.get("mfc", enc["color"]), mew=0.8, zorder=3)
    if r["verdict_pooled"] not in ("stable-a", "stable-b"):
        ax.text(r["upper"] + 1.2, yc - OFF - 0.25, "pooled: no winner", fontsize=style.ANNOT, color="0.35",
                ha="left", va="center")

# row labels and benchmark headers sit left of the axes, aligned on the label column
ax.set_yticks(ys)
ax.set_yticklabels([f"{short(a)} vs. {short(b)}" for a, b in zip(df["model_a"], df["model_b"])])
ax.tick_params(axis="y", length=0, pad=2, labelsize=7)
ax.spines["left"].set_visible(False)
LABEL_X = 0.01 / FIG_W  # figure fraction, left edge of the header text
for yh, name in heads:
    ax.text(LABEL_X, yh, name, transform=blended_transform_factory(fig.transFigure, ax.transData),
            fontsize=7.5, fontweight="bold", ha="left", va="center")

ax.xaxis.set_major_locator(FixedLocator([-20, -10, 0, 10, 20]))
ax.set_xticklabels(["−20", "−10", "0", "10", "20"])
ax.set_xlabel("gap: first minus second model (points)", labelpad=2)

handles = [Line2D([], [], color=style.COND[k]["color"], marker=style.COND[k]["marker"],
                  mfc=style.COND[k].get("mfc", style.COND[k]["color"]), mec=style.COND[k]["color"],
                  ms=3.8, lw=1.0, mew=0.8) for k in ("zero", "two", "pooled")]
fig.legend(handles, ["zero-shot", "two-shot", "pooled (average of both)"], loc="upper center",
           bbox_to_anchor=(0.5, 0.993), ncol=3, handlelength=1.6, columnspacing=1.0, handletextpad=0.4,
           borderaxespad=0.0)

out = style.save(fig, "fig2_reversals")
page = pymupdf.open(out)[0].rect
print(f"final size: {page.width / 72:.3f} x {page.height / 72:.3f} in")
check(abs(page.width / 72 - FIG_W) < 0.03 and abs(page.height / 72 - FIG_H) < 0.03,
      f"saved size matches {FIG_W} x {FIG_H} in")
