"""Figure F.2: shot-format control on ARC-Easy, 3.3 x 2.0 in.

Four models, identity option order. Per condition: mean accuracy over the four
wordings (marker) and the range over the wordings (thin bar). Conditions:
zero-shot, two-shot with the examples written inline (the protocol used
throughout), and two-shot with the same examples given as earlier chat turns.

Data: data/llm_decomp_{m}-Instruct.npy (wording x order x shot x item; order 0 is
the identity order) and data/llm_chatshot_{m}-Instruct.npy (wording x item).
Run from the repository root:
    python figscripts/fig3_shotformat.py
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

import style

FIG_W, FIG_H = 3.3, 2.0
MODELS = ["Llama-3.2-3B", "Llama-3.2-1B", "Qwen2.5-0.5B", "Falcon3-3B"]
CONDS = ["zero", "two", "chat"]

# Published per-wording accuracies (wordings 1-4, identity order), as printed in Table F.3.
EXPECTED = {
    "Llama-3.2-3B": {"zero": [.883, .881, .883, .882], "two": [.782, .752, .775, .366], "chat": [.895, .895, .893, .895]},
    "Llama-3.2-1B": {"zero": [.489, .679, .693, .671], "two": [.268, .322, .405, .343], "chat": [.741, .719, .735, .730]},
    "Qwen2.5-0.5B": {"zero": [.704, .653, .697, .683], "two": [.514, .463, .592, .573], "chat": [.731, .729, .731, .736]},
    "Falcon3-3B": {"zero": [.863, .871, .860, .864], "two": [.878, .878, .880, .876], "chat": [.881, .880, .879, .882]},
}


def check(cond, msg):
    if not cond:
        print("REPRODUCTION CHECK FAILED:", msg)
        sys.exit(1)
    print("check ok:", msg)


# ---------------------------------------------------------------- data
acc = {}
for m in MODELS:
    base = np.load(style.REPO / f"data/llm_decomp_{m}-Instruct.npy").astype(float)
    chat = np.load(style.REPO / f"data/llm_chatshot_{m}-Instruct.npy").astype(float)
    check(base.shape[:3] == (4, 4, 2) and chat.shape == (4, base.shape[3]),
          f"{m}: tensor shapes {base.shape} and {chat.shape}")
    acc[m] = {"zero": base[:, 0, 0, :].mean(1), "two": base[:, 0, 1, :].mean(1), "chat": chat.mean(1)}

# ---------------------------------------------------------------- reproduction checks
n_ok = 0
for m in MODELS:
    for c in CONDS:
        got = acc[m][c]
        print(f"{m:<13}{c:<5}", " ".join(f"{v:.4f}" for v in got), f" mean {got.mean():.4f}")
        ok = np.abs(got - np.array(EXPECTED[m][c])) <= 0.0005 + 1e-12
        n_ok += int(ok.sum())
check(n_ok == 48, f"{n_ok} of 48 per-wording values equal the published values at 3 decimals")
two3, chat3 = acc["Llama-3.2-3B"]["two"].mean(), acc["Llama-3.2-3B"]["chat"].mean()
check(round(two3, 3) == 0.669, f"Llama-3.2-3B inline mean {two3:.4f} is 0.669")
check(round(chat3, 4) == 0.8945 and round(chat3, 2) == 0.89, f"Llama-3.2-3B chat mean {chat3:.5f} is 0.8945 (0.89)")
w4 = acc["Llama-3.2-3B"]["two"][3]
check(f"{w4:.3f}" == "0.366" and w4 == acc["Llama-3.2-3B"]["two"].min(),
      "Llama-3.2-3B inline wording 4 is 0.366, the lowest wording")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(FIG_W, FIG_H))
X0, Y0, W, H = 0.63, 0.33, 2.67, 1.645   # axes box in inches
ax = fig.add_axes([X0 / FIG_W, Y0 / FIG_H, W / FIG_W, H / FIG_H])
OFF = {"zero": 0.25, "two": 0.0, "chat": -0.25}
ypos = {m: len(MODELS) - 1 - i for i, m in enumerate(MODELS)}
for m in MODELS:
    for c in CONDS:
        enc = style.COND[c]
        v, y = acc[m][c], ypos[m] + OFF[c]
        ax.plot([v.min(), v.max()], [y, y], color=enc["color"], lw=0.9, solid_capstyle="butt", zorder=2)
        ax.plot([v.mean()], [y], ls="none", marker=enc["marker"], ms=3.8, color=enc["color"],
                mec=enc["color"], mfc=enc.get("mfc", enc["color"]), mew=0.8, zorder=3)
# The wording-4 outlier that drives the Llama-3.2-3B collapse.
y = ypos["Llama-3.2-3B"] + OFF["two"]
ax.plot([w4, w4], [y - 0.09, y + 0.09], color=style.COND["two"]["color"], lw=0.9, zorder=2)
ax.text(w4 - 0.012, y, f"{w4:.3f}", fontsize=style.ANNOT, ha="right", va="center")

ax.set_ylim(-0.62, len(MODELS) - 0.45)
ax.set_yticks([ypos[m] for m in MODELS])
ax.set_yticklabels(MODELS)
ax.tick_params(axis="y", length=0, pad=3)
ax.spines["left"].set_visible(False)
ax.set_xlim(0.24, 0.92)
ax.xaxis.set_major_locator(FixedLocator([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
ax.set_xticklabels(["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
ax.set_xlabel("ARC-Easy accuracy, identity option order", labelpad=2)
for yy in np.arange(len(MODELS) - 1) + 0.5:
    ax.axhline(yy, color="0.9", lw=0.5, zorder=0)

handles = [Line2D([], [], color=style.COND[c]["color"], marker=style.COND[c]["marker"],
                  mfc=style.COND[c].get("mfc", style.COND[c]["color"]), mec=style.COND[c]["color"],
                  ms=3.8, lw=0.9, mew=0.8) for c in CONDS]
ax.legend(handles, [style.COND[c]["label"] for c in CONDS], loc="lower left", handlelength=1.6,
          labelspacing=0.15, handletextpad=0.4, borderaxespad=0.1)

out = style.save(fig, "fig3_shotformat")
page = pymupdf.open(out)[0].rect
print(f"final size: {page.width / 72:.3f} x {page.height / 72:.3f} in")
check(abs(page.width / 72 - FIG_W) < 0.03 and abs(page.height / 72 - FIG_H) < 0.03,
      f"saved size matches {FIG_W} x {FIG_H} in")
