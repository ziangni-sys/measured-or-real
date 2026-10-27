from pathlib import Path

import numpy as np
from scipy.stats import f as fdist

ROOT = Path(__file__).resolve().parents[1]

def icc1k_ci(reps, alpha=0.05):

    K, N = reps.shape
    mean = reps.mean(0)
    grand = mean.mean()
    msb = K * ((mean - grand) ** 2).sum() / (N - 1)
    msw = ((reps - mean) ** 2).sum() / (N * (K - 1))
    Fs = msb / msw
    d1, d2 = N - 1, N * (K - 1)
    fl = Fs / fdist.ppf(1 - alpha / 2, d1, d2)
    fu = Fs / fdist.ppf(alpha / 2, d1, d2)
    return 1 - 1 / Fs, 1 - 1 / fl, 1 - 1 / fu

def zc(Z):
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-8)

def per_dim_rel(Za, Zb):
    a, b = zc(Za), zc(Zb)
    num = (a * b).sum(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0)) + 1e-9
    return num / den

def procrustes_R(Zs, Z0):
    u, _, vt = np.linalg.svd(Zs.T @ Z0, full_matrices=False)
    return u @ vt

def main():

    du = np.stack([np.load(ROOT / "data" / "prune_scores" / f"seed{s}_e30.npz")["du"] for s in range(4)])
    el = np.stack([np.load(ROOT / "data" / "prune_scores" / f"seed{s}_e30.npz")["el"] for s in range(4)])

    for name, r in [("EL2N", el), ("Dyn-Unc", du)]:

        K, N = r.shape
        mean = r.mean(0); grand = mean.mean()
        msb = K * ((mean - grand) ** 2).sum() / (N - 1)
        msw = ((r - mean) ** 2).sum() / (N * (K - 1))
        icc1 = (msb - msw) / (msb + (K - 1) * msw)
        Fs = msb / msw; d1, d2 = N - 1, N * (K - 1)
        fl = Fs / fdist.ppf(0.975, d1, d2); fu = Fs / fdist.ppf(0.025, d1, d2)
        lo = (fl - 1) / (fl + K - 1); hi = (fu - 1) / (fu + K - 1)
        print(f"{name}: single-seed ICC(1,1)={icc1:.3f}  95% CI=[{lo:.3f},{hi:.3f}]  (N={N}, K={K})")

    FD = ROOT / "data" / "gauge_feats"
    if not all((FD / f"feat_seed{s}_e20.npy").exists() for s in range(3)):
        print("embedding part skipped: data/gauge_feats/ is absent; see KNOWN_LIMITATIONS.md")
        return
    Z = [np.load(FD / f"feat_seed{s}_e20.npy") for s in range(3)]
    n = Z[0].shape[0]
    rng = np.random.default_rng(0)
    idx = rng.permutation(n); fit, ev = idx[: n // 2], idx[n // 2:]
    pair_means = []
    for a in range(3):
        for b in range(a + 1, 3):
            Za, Zb = zc(Z[a]), zc(Z[b])
            R = procrustes_R(Za[fit], Zb[fit])
            pair_means.append(per_dim_rel((Za @ R)[ev], Zb[ev]).mean())
    pair_means = np.array(pair_means)

    Za, Zb = zc(Z[0]), zc(Z[1]); R = procrustes_R(Za[fit], Zb[fit])
    A, B = (Za @ R)[ev], Zb[ev]
    boot = []
    for _ in range(1000):
        bi = rng.integers(0, len(ev), len(ev))
        boot.append(per_dim_rel(A[bi], B[bi]).mean())
    boot = np.array(boot)
    print(f"embedding aligned reliability: mean over 3 seed-pairs={pair_means.mean():.3f} "
          f"range=[{pair_means.min():.3f},{pair_means.max():.3f}]")
    print(f"  image-bootstrap 95% CI (seed0-1 pair)=[{np.percentile(boot,2.5):.3f},{np.percentile(boot,97.5):.3f}]")

if __name__ == "__main__":
    main()
