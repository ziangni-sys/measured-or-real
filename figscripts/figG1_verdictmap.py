"""Figure G.1: zero-shot gap against two-shot gap for every pair.

One panel per benchmark. Each pair is oriented so that its zero-shot gap is
non-negative; agreeing pairs then sit in the upper-right quadrant and pairs
whose two protocols disagree in sign sit in the lower-right quadrant. Thin
crosshairs are the simultaneous intervals of Algorithm 2 in each protocol.

Data: ALG2/pairwise.csv (gap/lower/upper per protocol, verdict).
Run from the repository root:  python figscripts/figG1_verdictmap.py
"""

import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

import style

BENCH = [("arc", "ARC-Easy"), ("hellaswag", "HellaSwag"),
         ("mmlu", "MMLU"), ("openbookqa", "OpenBookQA")]
EXPECTED = {"arc": (93, 5, 7), "hellaswag": (42, 4, 9),
            "mmlu": (43, 3, 9), "openbookqa": (43, 4, 8)}
ORDER = ["certified", "protocol-dependent", "refused"]
LIMITS = {"arc": (80, (-30, 80)), "hellaswag": (40, (-20, 40)),
          "mmlu": (40, (-20, 40)), "openbookqa": (60, (-20, 60))}
XMIN = -3
TICK = 20  # same tick step (points) in every panel
MS = {"certified": 3.2, "protocol-dependent": 3.9, "refused": 3.0}


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        sys.exit(1)


def load():
    pw = pd.read_csv(style.ALG2 / "pairwise.csv")
    pw["v"] = pw["verdict"].map(style.VERDICT_MAP)
    check(pw["v"].notna().all(), "every verdict label maps to a plotted class")
    # Orient each pair so that the zero-shot gap is >= 0, in points.
    sign = np.where(pw["gap_protocol_0"] < 0, -1.0, 1.0)
    for p in (0, 1):
        g, lo, hi = (100 * pw[f"{k}_protocol_{p}"] for k in ("gap", "lower", "upper"))
        pw[f"g{p}"] = sign * g
        pw[f"lo{p}"] = np.where(sign > 0, lo, -hi)
        pw[f"hi{p}"] = np.where(sign > 0, hi, -lo)
    return pw


def checks(pw):
    for b, _ in BENCH:
        counts = tuple(int((pw[(pw.benchmark == b)].v == v).sum()) for v in ORDER)
        check(counts == EXPECTED[b], f"{b} certified/protocol-dependent/refused = "
                                     f"{counts[0]}/{counts[1]}/{counts[2]}")
    check(len(pw) == 270, f"pairs = {len(pw)}")
    check(int((pw.v == "certified").sum()) == 221, "221 certified in total")
    check(int((pw.v == "protocol-dependent").sum()) == 16, "16 protocol-dependent in total")
    # The verdict rule is reproduced from the plotted intervals.
    excl = [(pw[f"lo{p}"] > 0) | (pw[f"hi{p}"] < 0) for p in (0, 1)]
    same = np.sign(pw["g0"]) == np.sign(pw["g1"])
    rule = np.where(excl[0] & excl[1] & same, "certified",
                    np.where(excl[0] & excl[1], "protocol-dependent", "refused"))
    check(bool((rule == pw.v).all()), "verdicts follow from the plotted intervals")
    for _, r in pw.iterrows():
        xmax, (ymin, ymax) = LIMITS[r.benchmark]
        if not (XMIN <= r.lo0 and r.hi0 <= xmax and ymin <= r.lo1 and r.hi1 <= ymax):
            check(False, f"interval outside axis range: {r.benchmark} {r.model_a} {r.model_b}")
    check(True, "all intervals inside the axis ranges")


def handles():
    return [plt.Line2D([], [], ls="none", marker=style.VERDICT[v]["marker"], ms=3.5,
                       mfc=style.VERDICT[v].get("mfc", style.VERDICT[v]["color"]),
                       mec=style.VERDICT[v]["color"], mew=0.6) for v in ORDER]


def draw(pw):
    fig, axes = plt.subplots(1, 4, figsize=(style.TEXT_W, 2.1))
    for k, (ax, (b, title)) in enumerate(zip(axes, BENCH)):
        d = pw[pw.benchmark == b]
        xmax, (ymin, ymax) = LIMITS[b]
        ax.axvspan(0, xmax, ymin=0, ymax=(0 - ymin) / (ymax - ymin),
                   color="#D55E00", alpha=0.06, lw=0, zorder=0)
        ax.axhline(0, color="0.55", lw=0.5, zorder=1)
        ax.axvline(0, color="0.55", lw=0.5, zorder=1)
        t = np.array([0, xmax])
        ax.plot(t, t, color="0.75", lw=0.6, ls="--", zorder=1)
        for v in ORDER:
            dv = d[d.v == v]
            enc = style.VERDICT[v]
            ax.hlines(dv.g1, dv.lo0, dv.hi0, color=enc["color"], lw=0.45, alpha=0.7, zorder=2)
            ax.vlines(dv.g0, dv.lo1, dv.hi1, color=enc["color"], lw=0.45, alpha=0.7, zorder=2)
            filled = "mfc" not in enc
            ax.plot(dv.g0, dv.g1, ls="none", marker=enc["marker"], ms=MS[v],
                    mfc=enc.get("mfc", enc["color"]), mec="white" if filled else enc["color"],
                    mew=0.3 if filled else 0.6, zorder=3 if v == "certified" else 4)
        # Per-panel verdict counts, keyed by the verdict markers.
        counts = [int((d.v == v).sum()) for v in ORDER]
        ax.legend(handles(), [str(n) for n in counts], loc="lower right", ncol=3,
                  handletextpad=0.1, columnspacing=0.6, handlelength=1.0,
                  borderaxespad=0.2)
        ax.set_xlim(XMIN, xmax)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_locator(MultipleLocator(TICK))
        ax.yaxis.set_major_locator(MultipleLocator(TICK))
        ax.set_title(f"({'abcd'[k]}) {title}", loc="left", pad=7)
        ax.set_xlabel("zero-shot gap (points)")
        if k == 0:
            ax.set_ylabel("two-shot gap (points)")
    fig.tight_layout(pad=0.1, w_pad=1.2, rect=(0, 0, 1, 0.91))
    # Capitals at 6 pt stand in for the small caps the paper uses for verdict names.
    fig.legend(handles(), [v.upper() for v in ORDER], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.025), handletextpad=0.3, columnspacing=1.5,
               prop={"size": 6})
    return fig


def check_size(path, w, h, tol=0.04):
    import pymupdf
    with pymupdf.open(path) as doc:
        r = doc[0].rect
    ww, hh = r.width / 72, r.height / 72
    check(abs(ww - w) <= tol and abs(hh - h) <= tol, f"PDF size {ww:.2f} x {hh:.2f} in (target {w} x {h})")


def main():
    pw = load()
    checks(pw)
    out = style.save(draw(pw), "figG1_verdictmap")
    check_size(out, style.TEXT_W, 2.1)


if __name__ == "__main__":
    main()
