"""Figure F.1: where prompt variance comes from, and how reliably items are measured.

Two panels at 7.0 x 2.8 in, one row per ARC model (15 models, grouped by family).
(a) Shares of the variance of the 32 configuration-level ARC accuracies that
    belong to wording, option order, shot count and the residual.
(b) Item-difficulty G coefficients: g(1) for one configuration and g(32) for
    the mean of the 32 configurations (both protocols pooled), on ARC in black.
    Grey markers are the HellaSwag, MMLU and OpenBookQA cells of the 11 models
    compared on all four benchmarks.

Data: data/llm_decomp_{model}.npy (ARC) and
data/llm_decomp_{hellaswag,mmlu,openbookqa}_{model}.npy; var_shares and
gtheory from scripts/analyze_scale.py; model lists from the per-benchmark rows
of the reported pairwise.csv. Run from the repository root:
    python figscripts/figF1_gcoef.py
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style

sys.dont_write_bytecode = True  # importing the analysis module must not write into scripts/
sys.path.insert(0, str(style.REPO / "scripts"))
from analyze_scale import gtheory, var_shares  # noqa: E402

OFF_ARC = ["hellaswag", "mmlu", "openbookqa"]
FAMILIES = ["Qwen", "SmolLM2", "Falcon3", "Llama-3.2"]
SHARES = [("w", "wording", "#E69F00"), ("o", "option order", "#009E73"),
          ("s", "shot count", "#0072B2"), ("r", "residual", "#C8C8C8")]
GREY = "0.62"

# Table F.2 as printed (acc, spread, wording, order, shot, residual, g(1), g(32)).
TABLE_F2 = {
    "Qwen2.5-0.5B": (".610", "26", ".19", ".00", ".73", ".08", ".36", ".95"),
    "Qwen2-0.5B": (".613", "8", ".71", ".01", ".04", ".24", ".48", ".97"),
    "Qwen2.5-1.5B": (".886", "4", ".24", ".03", ".43", ".30", ".58", ".98"),
    "Qwen2-1.5B": (".845", "2", ".27", ".21", ".16", ".36", ".65", ".98"),
    "Qwen2.5-3B": (".927", "1.5", ".34", ".13", ".05", ".48", ".59", ".98"),
    "Qwen2.5-7B": (".962", "1.1", ".09", ".46", ".26", ".19", ".65", ".98"),
    "Qwen2-7B": (".946", "1.4", ".20", ".05", ".49", ".25", ".65", ".98"),
    "SmolLM2-135M": (".248", "4", ".04", ".63", ".01", ".32", ".04", ".54"),
    "SmolLM2-360M": (".255", "3", ".03", ".51", ".00", ".46", ".04", ".55"),
    "SmolLM2-1.7B": (".758", "13", ".42", ".09", ".06", ".42", ".43", ".96"),
    "Falcon3-1B": (".762", "4", ".05", ".04", ".81", ".10", ".51", ".97"),
    "Falcon3-3B": (".871", "4", ".14", ".21", ".48", ".17", ".55", ".97"),
    "Falcon3-7B": (".945", "1.6", ".07", ".69", ".01", ".23", ".62", ".98"),
    "Llama-3.2-1B": (".493", "45", ".18", ".00", ".79", ".03", ".15", ".85"),
    "Llama-3.2-3B": (".781", "52", ".29", ".00", ".41", ".29", ".34", ".94"),
}


def check(ok, msg):
    if not ok:
        sys.exit(f"REPRODUCTION CHECK FAILED: {msg}")


def fmt_like(x, printed):
    """Format x with the precision of the printed table entry."""
    dec = len(printed.split(".")[1]) if "." in printed else 0
    out = f"{x:.{dec}f}"
    return out[1:] if printed.startswith(".") and out.startswith("0.") else out


def family(m):
    return next(f for f in FAMILIES if m.startswith(f))


def size_key(m):
    """Order within a family as in Table F.2: Qwen by size then version, others by size."""
    tail = m.rsplit("-", 1)[1]
    val = float(tail[:-1]) * (1e-3 if tail.endswith("M") else 1.0)
    if family(m) == "Qwen":
        return (val, 0 if m.startswith("Qwen2.5") else 1)
    return (val, 0)


def model_order(models):
    out = []
    for f in FAMILIES:
        out += sorted([m for m in models if family(m) == f], key=size_key)
    return out


def main():
    pw = pd.read_csv(style.ALG2 / "pairwise.csv")
    names = {b: sorted({m.removesuffix("-Instruct") for m in pd.concat([g.model_a, g.model_b])})
             for b, g in pw.groupby("benchmark")}
    arc = model_order(names["arc"])
    eleven = names["hellaswag"]
    check(len(arc) == 15 and all(names[b] == eleven for b in OFF_ARC) and len(eleven) == 11,
          "model lists: 15 on ARC and the same 11 on the other benchmarks")
    check(arc == list(TABLE_F2), f"row order {arc}")

    rows, off = {}, {}
    for m in arc:
        C = np.load(style.REPO / "data" / f"llm_decomp_{m}-Instruct.npy")
        check(C.shape == (4, 4, 2, 2365), f"{m}: tensor shape {C.shape}")
        v = var_shares(C)
        g1, g32 = gtheory(C)
        rows[m] = dict(acc=v["mean"], spr=100 * (v["hi"] - v["lo"]), w=v["w"], o=v["o"],
                       s=v["s"], r=v["r"], g1=g1, g32=g32)
    for m in eleven:
        for b in OFF_ARC:
            C = np.load(style.REPO / "data" / f"llm_decomp_{b}_{m}-Instruct.npy")
            check(C.shape[:3] == (4, 4, 2) and C.shape[3] == 2365, f"{b} {m}: shape {C.shape}")
            off[(m, b)] = gtheory(C)

    # Reproduction checks: every Table F.2 row at printed precision, then the summaries.
    keys = ["acc", "spr", "w", "o", "s", "r", "g1", "g32"]
    for m in arc:
        got = tuple(fmt_like(rows[m][k], p) for k, p in zip(keys, TABLE_F2[m]))
        print(f"{m:13s} " + " ".join(got))
        check(got == TABLE_F2[m], f"Table F.2 row {m}: {got} != {TABLE_F2[m]}")
    max_arc = max(r["g1"] for r in rows.values())
    max_off = max(g1 for g1, _ in off.values())
    signal = [g32 for g1, g32 in off.values() if g1 >= 0.2]
    chance = [m for m in arc if m.startswith("SmolLM2") and m.endswith("M")]  # 135M, 360M
    smol = max([rows[m]["g32"] for m in chance]
               + [g32 for (m, _), (_, g32) in off.items() if m in chance])
    print(f"max g(1): ARC {max_arc:.3f}, off-ARC {max_off:.3f}; off-ARC cells with g(1) >= 0.2: "
          f"{len(signal)}, min g(32) {min(signal):.3f}; near-chance SmolLM2 max g(32) {smol:.3f}")
    check(round(max_arc, 3) == 0.651, "max ARC g(1) != 0.651")
    check(round(max_off, 3) == 0.622, "max off-ARC g(1) != 0.622")
    check(len(signal) == 24 and round(min(signal), 3) == 0.910, "24 cells with g(32) >= 0.910")
    check(round(smol, 3) == 0.638, "near-chance SmolLM2 max g(32) != 0.638")

    # Row positions: one unit per model, a half-unit gap between families.
    ypos, y = {}, 0.0
    for i, m in enumerate(arc):
        if i and family(m) != family(arc[i - 1]):
            y += 0.55
        ypos[m] = y
        y += 1.0
    yy = np.array([ypos[m] for m in arc])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(style.TEXT_W, 2.8), sharey=True,
                                   layout="constrained", gridspec_kw=dict(wspace=0.06))
    # Small outer pads keep every element on the canvas, which is saved at its exact size.
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)
    left = np.zeros(len(arc))
    for k, lab, col in SHARES:
        val = np.array([rows[m][k] for m in arc])
        axa.barh(yy, val, left=left, height=0.72, color=col, edgecolor="white", lw=0.5,
                 label=lab, zorder=2)
        left += val
    axa.set_xlim(0, 1)
    axa.set_xticks(np.linspace(0, 1, 6))
    axa.set_xticklabels([f"{t:g}" for t in np.linspace(0, 1, 6)])
    axa.set_xlabel("share of variance in configuration-level accuracy")
    axa.set_yticks(yy)
    axa.set_yticklabels(arc)
    axa.tick_params(axis="y", length=0, pad=3)
    axa.spines["left"].set_visible(False)
    axa.invert_yaxis()
    axa.set_title("(a) Where ARC-Easy accuracy varies across configurations", loc="left", pad=13)
    axa.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, handlelength=1.0,
               handleheight=0.8, columnspacing=1.2, handletextpad=0.4, borderaxespad=0.15,
               borderpad=0)

    # (b) dumbbells on ARC, with the off-ARC cells of the 11 models as grey markers just below.
    dy = 0.36
    for m in arc:
        r = rows[m]
        axb.plot([r["g1"], r["g32"]], [ypos[m]] * 2, color="black", lw=0.8, zorder=3)
        axb.plot(r["g1"], ypos[m], "o", ms=3.6, mfc="white", mec="black", mew=0.8, zorder=4)
        axb.plot(r["g32"], ypos[m], "o", ms=3.6, color="black", zorder=4)
        if m in eleven:
            g1s = [off[(m, b)][0] for b in OFF_ARC]
            g32s = [off[(m, b)][1] for b in OFF_ARC]
            axb.plot(g1s, [ypos[m] + dy] * 3, "o", ms=2.4, mfc="white", mec=GREY, mew=0.6,
                     ls="none", zorder=2)
            axb.plot(g32s, [ypos[m] + dy] * 3, "o", ms=2.4, color=GREY, ls="none", zorder=2)
    axb.set_xlim(-0.02, 1.0)
    axb.set_xticks(np.linspace(0, 1, 6))
    axb.set_xticklabels([f"{t:g}" for t in np.linspace(0, 1, 6)])
    axb.grid(axis="x", color="0.93", lw=0.5, zorder=0)
    axb.set_xlabel("generalizability coefficient $g$")
    axb.tick_params(axis="y", length=0)
    axb.spines["left"].set_visible(False)
    axb.set_title("(b) Item-difficulty reliability", loc="left", pad=13)
    handles = [Line2D([], [], marker="o", ls="none", ms=3.6, mfc="white", mec="black", mew=0.8),
               Line2D([], [], marker="o", ls="none", ms=3.6, color="black"),
               (Line2D([], [], marker="o", ls="none", ms=2.4, mfc="white", mec=GREY, mew=0.6),
                Line2D([], [], marker="o", ls="none", ms=2.4, color=GREY))]
    labels = ["$g(1)$: one configuration", "$g(32)$: mean of all 32", "other benchmarks"]
    axb.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
               handletextpad=0.1, columnspacing=1.1, borderaxespad=0.15, borderpad=0,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)})
    axb.set_ylim(yy.max() + 0.75, yy.min() - 0.6)
    style.save(fig, "figF1_gcoef", tight=False)


if __name__ == "__main__":
    main()
