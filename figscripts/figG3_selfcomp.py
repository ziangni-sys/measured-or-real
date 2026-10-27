"""Figure G.3: false stable winners in split-grid self-comparisons.

Each model is compared with itself on two disjoint halves of its prompt grid
(200 random splits per model), so any stable winner is an error. y = share of
splits with a false stable winner, for Algorithm 2 and for the pooled
comparison; x = the model's configuration spread on that benchmark (largest
minus smallest accuracy over its 32 configurations, in points, log scale).

Data: ALG2/cross_harness_native.json ({bench}.per_model_stable,
pooled_per_model_stable, repetitions_per_model); spread from the correctness
tensors data/llm_decomp_[{bench}_]{m}.npy (wording x order x shot x items).
Off ARC only the 11 reported models appear, as listed in the JSON.
Run from the repository root:  python figscripts/figG3_selfcomp.py
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, MultipleLocator, NullLocator

import style

BENCH = [("arc", "ARC-Easy", "o"), ("hellaswag", "HellaSwag", "s"),
         ("mmlu", "MMLU", "v"), ("openbookqa", "OpenBookQA", "P")]  # no diamond: it means "pooled" elsewhere
EXPECTED = {"arc": (157, 581, 3000), "hellaswag": (35, 292, 2200),
            "mmlu": (32, 254, 2200), "openbookqa": (118, 670, 2200)}
EXCLUDED_OFF_ARC = {"Qwen2-7B-Instruct", "Qwen2.5-7B-Instruct", "Falcon3-7B-Instruct",
                    "Qwen2.5-3B-Instruct"}


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        sys.exit(1)


def load():
    js = json.loads((style.ALG2 / "cross_harness_native.json").read_text(encoding="utf-8"))
    rows = []
    for b, _, _ in BENCH:
        c = js[b]
        reps = c["repetitions_per_model"]
        tag = "" if b == "arc" else f"{b}_"
        for m, k in c["per_model_stable"].items():
            acc = np.load(style.REPO / "data" / f"llm_decomp_{tag}{m}.npy").astype(np.float64).mean(-1)
            rows.append(dict(bench=b, model=m, reps=reps, alg2=k,
                             pooled=c["pooled_per_model_stable"][m],
                             spread=100 * (acc.max() - acc.min()), ncfg=acc.size))
    return js, pd.DataFrame(rows)


def checks(js, d):
    tot = [0, 0, 0]
    for b, _, _ in BENCH:
        db = d[d.bench == b]
        got = (int(db.alg2.sum()), int(db.pooled.sum()), int(db.reps.sum()))
        check(got == EXPECTED[b] and got[2] == js[b]["comparisons"]
              and got[0] == js[b]["stable"] and got[1] == js[b]["pooled_stable"],
              f"{b}: Algorithm 2 {got[0]}, pooled {got[1]} of {got[2]}")
        tot = [t + g for t, g in zip(tot, got)]
    check(tot == [342, 1797, 9600], f"totals: Algorithm 2 {tot[0]}, pooled {tot[1]} of {tot[2]}")
    check(len(d) == 48, f"model-benchmark cells = {len(d)}")
    off = d[d.bench != "arc"]
    check(not (set(off.model) & EXCLUDED_OFF_ARC), "off-ARC cells use only the 11 reported models")
    check(bool((d.ncfg == 32).all()), "spread is taken over 32 configurations")
    check(bool(d.spread.between(*XLIM).all()), f"spread range {d.spread.min():.1f}-{d.spread.max():.1f} inside axes")


XLIM = (0.9, 70)
MS = 4.2  # marker size, pt, large enough to tell the four benchmark shapes apart


def draw(d):
    fig, ax = plt.subplots(figsize=(style.COL_W, 2.4))
    for b, _, mk in BENCH:
        db = d[d.bench == b]
        x = db.spread.to_numpy()
        ya, yp = (100 * db.alg2 / db.reps).to_numpy(), (100 * db.pooled / db.reps).to_numpy()
        ax.vlines(x, ya, yp, color="0.82", lw=0.6, zorder=1)
        for key, y in (("pooled", yp), ("ours", ya)):
            c = style.METHOD[key]["color"]
            ax.plot(x, y, ls="none", marker=mk, ms=MS, mfc=c, mec="white", mew=0.35,
                    zorder=3 if key == "ours" else 2)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(-3, 80)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel("configuration spread (points, log scale)")
    ax.set_ylabel("false stable winners (% of splits)")
    meth = [plt.Line2D([], [], ls="none", marker="o", ms=MS, mfc=style.METHOD[k]["color"],
                       mec="white", mew=0.35) for k in ("ours", "pooled")]
    leg1 = ax.legend(meth, ["Algorithm 2", "pooled comparison"], loc="upper left",
                     bbox_to_anchor=(0.0, 1.0), handletextpad=0.2, borderaxespad=0.2,
                     labelspacing=0.3, handlelength=1.0)
    ax.add_artist(leg1)
    bh = [plt.Line2D([], [], ls="none", marker=mk, ms=MS, mfc="0.45", mec="white", mew=0.35)
          for _, _, mk in BENCH]
    ax.legend(bh, [t for _, t, _ in BENCH], loc="upper left", bbox_to_anchor=(0.0, 0.80),
              handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3, handlelength=1.0)
    fig.tight_layout(pad=0.1)
    return fig


def check_size(path, w, h, tol=0.04):
    import pymupdf
    with pymupdf.open(path) as doc:
        r = doc[0].rect
    ww, hh = r.width / 72, r.height / 72
    check(abs(ww - w) <= tol and abs(hh - h) <= tol, f"PDF size {ww:.2f} x {hh:.2f} in (target {w} x {h})")


def main():
    js, d = load()
    checks(js, d)
    out = style.save(draw(d), "figG3_selfcomp")
    check_size(out, style.COL_W, 2.4)


if __name__ == "__main__":
    main()
