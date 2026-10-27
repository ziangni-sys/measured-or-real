import os
from pathlib import Path

import numpy as np
from scipy.stats import f as fdist

ROOT = Path(__file__).resolve().parents[1]
DATASET = os.environ.get("DATASET", "cifar100")
FDIR = ROOT / "data" / ("prune_scores" if DATASET == "cifar100" else f"prune_scores_{DATASET}")

def icc1k(M):
    k, n = M.shape
    mean = M.mean(0)
    grand = mean.mean()
    msb = k * ((mean - grand) ** 2).sum() / (n - 1)
    msw = ((M - mean[None, :]) ** 2).sum() / (n * (k - 1))
    return 1 - msw / msb if msb > 0 else 0.0

def pear(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0

def alg1_ucb(r, rx, ry, n, k, a=0.1):
    d1, d2 = n * (k - 1), n - 1
    qg = fdist.ppf(1 - a, d2, d1)
    if 1 / max(1 - rx, 1e-9) <= qg or 1 / max(1 - ry, 1e-9) <= qg:
        return float(np.clip(r, -1, 1)), True
    qu = fdist.ppf(1 - a, d1, d2)
    ux = max(1 - (1 - rx) / qu, 1e-3)
    uy = max(1 - (1 - ry) / qu, 1e-3)
    return float(np.clip(r / np.sqrt(ux * uy), -1, 1)), False

def naive_pt(r, rx, ry):
    return r / np.sqrt(max(rx * ry, 1e-9))

def load():
    el = np.stack([np.load(FDIR / f"seed{s}_e30.npz")["el"] for s in range(4)])
    du = np.stack([np.load(FDIR / f"seed{s}_e30.npz")["du"] for s in range(4)])
    return el, du

def icc11(M):
    k, n = M.shape
    mean = M.mean(0)
    msb = k * ((mean - mean.mean()) ** 2).sum() / (n - 1)
    msw = ((M - mean[None, :]) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw) if msb > 0 else 0.0

def pooled(el, du):
    n = el.shape[1]

    ex, ey = el - el.mean(0), du - du.mean(0)
    err = pear(ex.ravel(), ey.ravel())
    print(f"[{DATASET}] pooled N={n}: corr(eps_X,eps_Y) same-seed = {err:+.3f} "
          f"(nonzero => standard disattenuation biased; use cross-seed r_obs)")

    rx2, ry2 = icc1k(el[[0, 1]]), icc1k(du[[0, 1]])
    r_same = pear(el[[2, 3]].mean(0), du[[2, 3]].mean(0))
    u_same, _ = alg1_ucb(r_same, rx2, ry2, n, 2)
    print(f"  same-seed  (biased): r_obs={r_same:.3f}, rel {rx2:.3f}/{ry2:.3f} -> UCB={u_same:.3f}")

    sx, sy = icc11(el[[0, 1]]), icc11(du[[0, 1]])
    for a, b in [(2, 3), (3, 2)]:
        r_cross = pear(el[a], du[b])
        nv = naive_pt(r_cross, sx, sy)
        u, _ = alg1_ucb(r_cross, sx, sy, n, 2)
        print(f"  cross-seed X{a}/Y{b} (valid): r_obs={r_cross:.3f}, single-seed rel "
              f"{sx:.3f}/{sy:.3f} -> naive={nv:.3f} UCB={u:.3f}")

def per_class(el, du, labels):

    classes = np.unique(labels)
    ref, ind_ucb, ind_naive, over = [], [], [], 0
    for c in classes:
        m = labels == c
        nc = int(m.sum())
        elc, duc = el[:, m], du[:, m]
        rx4, ry4 = icc1k(elc), icc1k(duc)
        if min(rx4, ry4) <= 0:
            continue
        rc = pear(elc.mean(0), duc.mean(0)) / np.sqrt(rx4 * ry4)
        rx, ry = icc1k(elc[[0, 1]]), icc1k(duc[[0, 1]])
        robs = pear(elc[[2, 3]].mean(0), duc[[2, 3]].mean(0))
        nv = naive_pt(robs, rx, ry)
        u, _ = alg1_ucb(robs, rx, ry, nc, 2)
        ref.append(np.clip(rc, -1, 1))
        ind_ucb.append(u)
        ind_naive.append(nv)
        if abs(nv) > 1:
            over += 1
    ref, ind_ucb, ind_naive = np.array(ref), np.array(ind_ucb), np.array(ind_naive)
    print(f"[{DATASET}] per-class ({len(ref)} classes): "
          f"corr(indep-UCB, K=4 ref)={pear(ind_ucb, ref):.3f} "
          f"MAE={np.abs(ind_ucb - ref).mean():.3f}; "
          f"naive |r|>1 on {over}/{len(ref)} classes (UCB bounded on all)")
    return ref, ind_ucb, ind_naive

def main():
    el, du = load()
    pooled(el, du)
    lab = np.load(FDIR / "seed0_e30.npz").get("y")
    if lab is not None:
        per_class(el, du, lab)

if __name__ == "__main__":
    main()
