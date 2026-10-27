import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORDINGS, ORDERS, SHOTS = 4, 4, 2


def load(tag):
    return np.load(ROOT / "data" / f"llm_decomp_{tag}.npy")


def var_shares(C):
    acc = C.mean(3)
    gm = acc.mean()
    aw = acc.mean((1, 2)) - gm
    ao = acc.mean((0, 2)) - gm
    ash = acc.mean((0, 1)) - gm
    ss_w = ORDERS * SHOTS * (aw ** 2).sum()
    ss_o = WORDINGS * SHOTS * (ao ** 2).sum()
    ss_s = WORDINGS * ORDERS * (ash ** 2).sum()
    ss_tot = ((acc - gm) ** 2).sum()
    ss_res = ss_tot - ss_w - ss_o - ss_s
    return dict(mean=gm, std=acc.std(), lo=acc.min(), hi=acc.max(),
                w=ss_w / ss_tot, o=ss_o / ss_tot, s=ss_s / ss_tot, r=ss_res / ss_tot)


def gtheory(C):
    X = C.reshape(-1, C.shape[-1]).astype(float).T
    p, k = X.shape
    mi = X.mean(1, keepdims=True)
    mc = X.mean(0, keepdims=True)
    gm = X.mean()
    ss_i = k * ((mi - gm) ** 2).sum()
    ss_c = p * ((mc - gm) ** 2).sum()
    ss_res = ((X - mi - mc + gm) ** 2).sum()
    ms_i = ss_i / (p - 1)
    ms_c = ss_c / (k - 1)
    ms_res = ss_res / ((p - 1) * (k - 1))
    v_i = max((ms_i - ms_res) / k, 0.0)
    v_res = ms_res
    g1 = v_i / (v_i + v_res)
    gk = v_i / (v_i + v_res / k)
    return g1, gk


def var_shares_ci(C, B=1000):
    N = C.shape[-1]
    rng = np.random.default_rng(0)
    out = np.empty((B, 4))
    for b in range(B):
        idx = rng.integers(0, N, N)
        v = var_shares(C[:, :, :, idx])
        out[b] = [v["w"], v["o"], v["s"], v["r"]]
    return np.percentile(out, [2.5, 97.5], axis=0)


def gtheory_ci(C, B=2000):
    X = C.reshape(-1, C.shape[-1]).astype(float).T
    p, k = X.shape
    rng = np.random.default_rng(0)
    g1 = np.empty(B)
    for b in range(B):
        Xi = X[rng.integers(0, p, p)]
        mi = Xi.mean(1, keepdims=True)
        mc = Xi.mean(0, keepdims=True)
        gm = Xi.mean()
        ms_i = (k * ((mi - gm) ** 2).sum()) / (p - 1)
        ms_res = ((Xi - mi - mc + gm) ** 2).sum() / ((p - 1) * (k - 1))
        v_i = max((ms_i - ms_res) / k, 0.0)
        g1[b] = v_i / (v_i + ms_res)
    return np.percentile(g1, [2.5, 97.5])


def shot_split(C):
    acc = C.mean(3)
    return acc[:, :, 0].mean(), acc[:, :, 1].mean()


def pair(tagA, tagB, name):
    A, B = load(tagA), load(tagB)
    accA, accB = A.mean(3), B.mean(3)
    n = A.shape[-1]
    gap = (accA.mean() - accB.mean()) * 100
    dcfg = (accA - accB).reshape(-1)
    overall = 1 if gap > 0 else -1
    flips = int(np.sum(np.sign(dcfg) != overall) - np.sum(dcfg == 0) * (overall == 1))
    flips = int(np.sum((np.sign(dcfg) == -overall)))
    g0 = (accA[:, :, 0].mean() - accB[:, :, 0].mean()) * 100
    gf = (accA[:, :, 1].mean() - accB[:, :, 1].mean()) * 100
    d0 = (accA[:, :, 0] - accB[:, :, 0]).reshape(-1)
    df = (accA[:, :, 1] - accB[:, :, 1]).reshape(-1)
    fl0 = int(np.sum(np.sign(d0) == -np.sign(g0)))
    flf = int(np.sum(np.sign(df) == -np.sign(gf)))
    XA = A.reshape(-1, n).mean(0)
    XB = B.reshape(-1, n).mean(0)
    di = XA - XB
    rng = np.random.default_rng(0)
    boot = np.array([di[rng.integers(0, n, n)].mean() for _ in range(5000)]) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    prA = float((boot > 0).mean())
    print(f"\n### PAIR {name}  (N={n} items, 32 configs)")
    print(f"  pooled gap      = {gap:+.2f} pts   winner flips {flips}/32")
    print(f"  0-shot gap      = {g0:+.2f} pts   flips {fl0}/16")
    print(f"  few-shot gap    = {gf:+.2f} pts   flips {flf}/16")
    print(f"  item-boot 95%CI = [{lo:+.2f}, {hi:+.2f}] pts   Pr(A>B)={prA:.2f}")


def one(tag):
    C = load(tag)
    v = var_shares(C)
    g1, gk = gtheory(C)
    lo, hi = gtheory_ci(C)
    s0, sf = shot_split(C)
    print(f"\n### {tag}  (N={C.shape[-1]})")
    print(f"  acc mean={v['mean']:.3f} std={v['std']:.3f} range=[{v['lo']:.3f},{v['hi']:.3f}]  spread={100*(v['hi']-v['lo']):.0f}pts")
    sc = var_shares_ci(C)
    print(f"  var shares: wording={v['w']:.2f} order={v['o']:.2f} few-shot={v['s']:.2f} resid={v['r']:.2f}"
          f"  (95% CI few-shot [{sc[0,2]:.2f},{sc[1,2]:.2f}] order [{sc[0,1]:.2f},{sc[1,1]:.2f}])")
    print(f"  item-difficulty G: single-prompt g(1)={g1:.2f} [95% CI {lo:.2f},{hi:.2f}]  32-prompt g(32)={gk:.2f}")
    print(f"  0-shot acc={s0:.3f}  few-shot acc={sf:.3f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "1.5B"
    if which == "0.5B":
        one("Qwen2.5-0.5B-Instruct"); one("Qwen2-0.5B-Instruct")
        pair("Qwen2.5-0.5B-Instruct", "Qwen2-0.5B-Instruct", "0.5B")
    elif which == "1.5B":
        one("Qwen2.5-1.5B-Instruct"); one("Qwen2-1.5B-Instruct")
        pair("Qwen2.5-1.5B-Instruct", "Qwen2-1.5B-Instruct", "1.5B")
    elif which == "7B":
        one("Qwen2.5-7B-Instruct"); one("Qwen2-7B-Instruct")
        pair("Qwen2.5-7B-Instruct", "Qwen2-7B-Instruct", "7B")
