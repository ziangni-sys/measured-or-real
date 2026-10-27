"""Shared figure style and data paths for the paper's figures.

Two-column geometry: column width 3.3 in, text width 7.0 in. Figures are
drawn at final size and included at 100%, so 7-8pt text here matches the
10pt body. Each method, protocol and verdict keeps one encoding everywhere.

Paths follow the repository layout, with figscripts/ in the repository root,
so the scripts run from any working directory. Three environment variables
override the defaults: FIG_REPO (the folder that holds data/, results/ and
scripts/), FIG_OUT (where the PDFs go, default figures/) and FIG_PREVIEW
(where the PNG previews go, default figures/preview/).
"""

import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})

REPO = Path(os.environ.get("FIG_REPO") or Path(__file__).resolve().parents[1])
ALG2 = REPO / "results/algo2_reported"
ROBUST = REPO / "results/disatt_robustness_full_4000x500_seed20270720/results.csv"
OUT = Path(os.environ.get("FIG_OUT") or REPO / "figures")
PREVIEW = Path(os.environ.get("FIG_PREVIEW") or OUT / "preview")
if not (REPO / "data").is_dir() or not ALG2.is_dir():
    raise SystemExit(f"{REPO} has no data/ or results/algo2_reported/; "
                     "set FIG_REPO to the repository root.")
COL_W, TEXT_W = 3.3, 7.0
ANNOT = 7  # annotation font size, pt

# Fixed method encodings: the same method always has the same color, marker and line.
METHOD = {
    "ours":      dict(color="#0072B2", marker="^", ls="-",  lw=1.2, label="Algorithm 1"),
    "plugin":    dict(color="#D55E00", marker="o", ls="-",  lw=0.9, label="plug-in (floor $10^{-9}$)"),
    "clipped":   dict(color="#E69F00", marker="s", ls="-",  lw=0.9, label="clipped plug-in"),
    "floor":     dict(color="#7F7F7F", marker="D", ls="--", lw=0.9, label="fixed floor 0.1"),
    "bootstrap": dict(color="#009E73", marker="h", ls=":",  lw=0.9, label="bootstrap version"),
    "nogate":    dict(color="#56B4E9", marker="v", ls="-.", lw=0.9, label="upper bound, no gate"),
    "gatepoint": dict(color="#000000", marker="x", ls=":",  lw=0.9, label="gate + point ICC"),
    "pooled":    dict(color="#CC79A7", marker="D", ls="-",  lw=0.9, label="pooled comparison"),
}
CSV_METHOD = {"algorithm1": "ours", "naive_plugin": "plugin", "clipped_plugin": "clipped",
              "clipping_only": "clipped", "fixed_floor_0.1": "floor", "bootstrap_ucb": "bootstrap",
              "ucb_no_gate": "nogate", "gate_point": "gatepoint"}
# LLM conditions (never in the same figure as "nogate"):
COND = {"zero": dict(color="#56B4E9", marker="s", mfc="white", label="zero-shot"),
        "two": dict(color="#0072B2", marker="s", label="two-shot (inline)"),
        "chat": dict(color="#009E73", marker="o", label="two-shot (chat turns)"),
        "pooled": dict(color="#CC79A7", marker="D", label="pooled")}
VERDICT = {"certified": dict(color="#0072B2", marker="o"),
           "protocol-dependent": dict(color="#D55E00", marker="^"),
           "refused": dict(color="#999999", marker="o", mfc="white")}
VERDICT_MAP = {"stable-a": "certified", "stable-b": "certified",
               "protocol-dependent": "protocol-dependent",
               "unresolved": "refused", "unresolved-mixed": "refused"}


def save(fig, name, tight=True):
    """Write the vector PDF used by LaTeX and a PNG preview for visual checks.

    tight=True crops the PDF to its content. tight=False keeps the exact canvas
    size, for a figure whose layout already keeps every element on the canvas.
    The PDF carries no creation date, so a rerun writes the same bytes.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}.pdf"
    with mpl.rc_context({"savefig.bbox": "tight" if tight else "standard"}):
        fig.savefig(out, metadata={"CreationDate": None})
        fig.savefig(PREVIEW / f"{name}.png", dpi=220)
    plt.close(fig)
    print(out)
    return out
