"""Figure G.2: certified ARC directions on four new wordings.

x = per-protocol gap on the original prompt grid (Algorithm 2, ARC rows of
ALG2/pairwise.csv); y = the same gap on the four wordings written after all
verdicts were fixed (mean over the new wordings at the identity option order,
from data/llm_holdout_{m}.npy, shape 4 new wordings x 2 shots x items).
Certified pairs are oriented so the certified winner is named first; pairs
with a protocol-dependent verdict are oriented so the zero-shot gap is >= 0.
A point in a shaded quadrant has a different sign on the new wordings.

The held-out logic re-implements scripts/holdout_wordings.py::analyze, which
is not run here because it rewrites data/holdout_wordings.json.
Run from the repository root:  python figscripts/figG2_newwordings.py
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MultipleLocator

import style

FAIL_PAIR = ("Qwen2.5-0.5B-Instruct", "Llama-3.2-1B-Instruct")


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        sys.exit(1)


def short(m):
    return m.replace("-Instruct", "")


def load():
    pw = pd.read_csv(style.ALG2 / "pairwise.csv")
    pw = pw[pw.benchmark == "arc"].reset_index(drop=True)
    models = sorted(set(pw.model_a) | set(pw.model_b))
    H = {}
    for m in models:
        C = np.load(style.REPO / "data" / f"llm_holdout_{m}.npy")
        check(C.shape[:2] == (4, 2) and (C >= 0).all(), f"held-out tensor complete: {short(m)}")
        H[m] = C.astype(np.float64)
    return pw, H


def analyze(pw, H):
    """Per pair and protocol: in-pool gap, held-out gap (points), and flags."""
    rows, cfg_flips, cfg_total = [], 0, 0
    for _, r in pw.iterrows():
        v = style.VERDICT_MAP[r.verdict]
        if v == "refused":
            continue
        a, b = r.model_a, r.model_b
        held = [(H[a][:, s, :] - H[b][:, s, :]).mean() for s in range(2)]
        inpool = [r.gap_protocol_0, r.gap_protocol_1]
        if v == "certified":
            sign = 1.0 if r.verdict == "stable-a" else -1.0
            per_cfg = (H[a].mean(2) - H[b].mean(2)).reshape(-1)
            cfg_flips += int((np.sign(per_cfg) == -sign).sum())
            cfg_total += per_cfg.size
            fails = any(np.sign(h) != sign for h in held)
            replicated = None
        else:
            sign = 1.0 if inpool[0] >= 0 else -1.0
            fails = None
            replicated = all(np.sign(h) == np.sign(g) for h, g in zip(held, inpool))
        for p in (0, 1):
            rows.append(dict(a=a, b=b, v=v, proto=p, x=100 * sign * inpool[p],
                             y=100 * sign * held[p], held_ab=100 * held[p],
                             fails=fails, replicated=replicated))
    return pd.DataFrame(rows), cfg_flips, cfg_total


def checks(d, cfg_flips, cfg_total):
    cert = d[d.v == "certified"]
    n_cert = len(cert) // 2
    check(n_cert == 93, f"certified ARC pairs checked = {n_cert}")
    failed = cert[cert.fails.astype(bool)].drop_duplicates(["a", "b"])
    check(len(failed) == 1, f"direction failures = {len(failed)}")
    f = failed.iloc[0]
    hg = cert[(cert.a == f.a) & (cert.b == f.b)].sort_values("proto").held_ab.round(1).tolist()
    check((f.a, f.b) == FAIL_PAIR and hg == [-2.6, 6.2],
          f"failing pair {short(f.a)} vs. {short(f.b)}, held-out gaps {hg[0]:+.1f}/{hg[1]:+.1f}")
    check((cfg_flips, cfg_total) == (5, 744), f"new-wording configuration flips = {cfg_flips}/{cfg_total}")
    pdp = d[d.v == "protocol-dependent"].drop_duplicates(["a", "b"])
    rep = int(pdp.replicated.astype(bool).sum())
    check((rep, len(pdp)) == (5, 5), f"protocol-dependent patterns replicated = {rep}/{len(pdp)}")
    # Plotted points must fall inside the axis range.
    check(bool(d.x.between(*XLIM).all() and d.y.between(*YLIM).all()), "all points inside the axes")


XLIM, YLIM = (-24, 76), (-20, 76)


def draw(d):
    fig, ax = plt.subplots(figsize=(style.COL_W, 3.0))
    shade = dict(color="#D55E00", alpha=0.06, lw=0, zorder=0)
    ax.fill_between([0, XLIM[1]], YLIM[0], 0, **shade)
    ax.fill_between([XLIM[0], 0], 0, YLIM[1], **shade)
    ax.axhline(0, color="0.55", lw=0.5, zorder=1)
    ax.axvline(0, color="0.55", lw=0.5, zorder=1)
    ax.plot(XLIM, XLIM, color="0.75", lw=0.6, ls="--", zorder=1)
    proto = {0: "zero", 1: "two"}
    for p, key in proto.items():
        enc = style.COND[key]
        for v, mk, ms, z in (("certified", enc["marker"], 3.2, 3), ("protocol-dependent", "^", 4.0, 4)):
            dv = d[(d.proto == p) & (d.v == v)]
            filled = "mfc" not in enc
            ax.plot(dv.x, dv.y, ls="none", marker=mk, ms=ms + (0.2 if filled else 0),
                    mec="white" if filled else enc["color"], mfc=enc.get("mfc", enc["color"]),
                    mew=0.3 if filled else 0.7, zorder=z)
    # Highlight both protocol points of the one pair that fails on the new wordings.
    fp = d[(d.a == FAIL_PAIR[0]) & (d.b == FAIL_PAIR[1])].sort_values("proto")
    ax.plot(fp.x, fp.y, ls="none", marker="o", ms=8.5, mfc="none", mec="black", mew=0.7, zorder=5)
    ax.plot(fp.x, fp.y, color="black", lw=0.5, zorder=2)
    fx, fy = fp.iloc[0].x, fp.iloc[0].y
    ax.annotate(f"{short(FAIL_PAIR[0])} vs.\n{short(FAIL_PAIR[1])}:\nfails at zero-shot",
                xy=(fx + 1.4, fy - 1.0), xytext=(fx + 22, fy - 8), fontsize=style.ANNOT,
                ha="left", va="center", linespacing=1.05,
                arrowprops=dict(arrowstyle="-", color="black", lw=0.5, shrinkA=1, shrinkB=2))
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel("gap on the original prompt grid (points)")
    ax.set_ylabel("gap on the four new wordings (points)")
    def mk(key, marker, ms):
        e = style.COND[key]
        return plt.Line2D([], [], ls="none", marker=marker, ms=ms, mec=e["color"],
                          mfc=e.get("mfc", e["color"]), mew=0.7)
    h = [mk("zero", "s", 3.5), mk("two", "s", 3.5), (mk("zero", "^", 4.0), mk("two", "^", 4.0))]
    # Legend right of x=0, in the empty area above the points and clear of the shading.
    ax.legend(h, ["zero-shot", "two-shot", "protocol-dependent pair"], loc="upper left",
              bbox_to_anchor=(0.26, 1.0), handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
              handletextpad=0.4, borderaxespad=0.2, labelspacing=0.3, handlelength=1.6)
    fig.tight_layout(pad=0.1)
    return fig


def check_size(path, w, h, tol=0.04):
    import pymupdf
    with pymupdf.open(path) as doc:
        r = doc[0].rect
    ww, hh = r.width / 72, r.height / 72
    check(abs(ww - w) <= tol and abs(hh - h) <= tol, f"PDF size {ww:.2f} x {hh:.2f} in (target {w} x {h})")


def main():
    pw, H = load()
    d, cfg_flips, cfg_total = analyze(pw, H)
    checks(d, cfg_flips, cfg_total)
    out = style.save(draw(d), "figG2_newwordings")
    check_size(out, style.COL_W, 3.0)


if __name__ == "__main__":
    main()
