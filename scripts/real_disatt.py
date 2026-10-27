from pathlib import Path

import numpy as np
from scipy.stats import f as fdist

ROOT = Path(__file__).resolve().parents[1]
SDIR = ROOT / "data" / "prune_scores"
K = 2
ALPHA = 0.25

def icc(reps):
    mean = reps.mean(0)
    grand = mean.mean()
    n = reps.shape[1]
    msb = reps.shape[0] * ((mean - grand) ** 2).sum() / (n - 1)
    msw = ((reps - mean) ** 2).sum() / (n * (reps.shape[0] - 1))
    return 1.0 - msw / msb if msb > 0 else 0.0

def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0

def naive(r, rx, ry):
    return r / np.sqrt(max(rx * ry, 1e-9))

def ctdf_ucb(r, rx, ry, n, k, alpha=ALPHA):
    d1, d2 = n * (k - 1), n - 1
    q_gate = fdist.ppf(1 - alpha, d2, d1)
    if 1 / max(1 - rx, 1e-9) <= q_gate or 1 / max(1 - ry, 1e-9) <= q_gate:
        return float(np.clip(r, -1, 1))
    q = fdist.ppf(1 - alpha, d1, d2)
    ux = max(1 - (1 - rx) / q, 1e-3)
    uy = max(1 - (1 - ry) / q, 1e-3)
    return float(np.clip(r / np.sqrt(ux * uy), -1, 1))

def main():
    du = np.stack([np.load(SDIR / f"seed{s}_e30.npz")["du"] for s in range(K)])
    el = np.stack([np.load(SDIR / f"seed{s}_e30.npz")["el"] for s in range(K)])
    label_path = ROOT / "data" / "cifar100_labels.npy"
    if label_path.exists():
        y = np.load(label_path)
    else:
        candidates = [ROOT / "data" / name / "train.npz" for name in ("cifar100_hf", "cifar_hf")]
        cached = next((path for path in candidates if path.exists()), None)
        if cached is None:
            raise FileNotFoundError(
                "missing data/cifar100_labels.npy; see KNOWN_LIMITATIONS.md"
            )
        y = np.load(cached)["labels"]
    classes = np.unique(y)

    rows = []
    for c in classes:
        idx = np.where(y == c)[0]
        n = len(idx)
        elc, duc = el[:, idx], du[:, idx]
        rx, ry = icc(elc), icc(duc)
        r_obs = corr(elc.mean(0), duc.mean(0))
        rows.append((r_obs, rx, ry, n))

    r_obs = np.array([r[0] for r in rows])
    rx = np.array([r[1] for r in rows])
    ry = np.array([r[2] for r in rows])
    nv = np.array([naive(*row[:3]) for row in rows])
    ug = np.array([ctdf_ucb(row[0], row[1], row[2], row[3], K) for row in rows])
    refused = np.array([1 / max(1 - row[1], 1e-9) <= fdist.ppf(1 - ALPHA, row[3] - 1, row[3] * (K - 1))
                        or 1 / max(1 - row[2], 1e-9) <= fdist.ppf(1 - ALPHA, row[3] - 1, row[3] * (K - 1))
                        for row in rows])

    print(f"=== REAL disattenuation: EL2N vs Dyn-Unc correlation per CIFAR-100 class ===")
    print(f"classes={len(classes)}  K={K} seeds  mean examples/class={int(np.mean([r[3] for r in rows]))}")
    print(f"reliability (ICC) EL2N: median={np.median(rx):.2f} range=[{rx.min():.2f},{rx.max():.2f}]")
    print(f"reliability (ICC) DynUnc: median={np.median(ry):.2f} range=[{ry.min():.2f},{ry.max():.2f}]")
    print(f"observed corr:  median={np.median(r_obs):+.2f}")
    print(f"naive disatt:   median={np.median(nv):+.2f}  |r|>1 on {int((np.abs(nv)>1.0).sum())}/{len(classes)} classes (impossible)")
    print(f"Alg.1 (UCB):    median={np.median(ug):+.2f}  |r|>1 on {int((np.abs(ug)>1.0).sum())}/{len(classes)}  refused={int(refused.sum())}/{len(classes)}")
    print(f"max |naive|={np.abs(nv).max():.2f}  max |Alg.1|={np.abs(ug).max():.2f}")

if __name__ == "__main__":
    main()
