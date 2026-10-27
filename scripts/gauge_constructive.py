from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
FDIR = ROOT / "data" / "gauge_feats"
EPOCHS = 20
SEEDS = 3

def zc(Z):
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-8)

def procrustes_to(Zs, Z0):
    u, _, vt = np.linalg.svd(Zs.T @ Z0, full_matrices=False)
    return Zs @ (u @ vt)

def probe_acc(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    return float((clf.predict(Xte) == yte).mean())

def per_dim_rel(stack):
    S = stack.shape[0]
    m = stack.mean(0)
    sig2 = stack.var(0, ddof=1).mean(0)
    var_sig = np.maximum(m.var(0, ddof=1) - sig2 / S, 1e-6)
    return var_sig / (var_sig + sig2 / S)

def main():
    Z = [np.load(FDIR / f"feat_seed{s}_e{EPOCHS}.npy") for s in range(SEEDS)]
    candidates = [ROOT / "data" / name / "test.npz" for name in ("cifar100_hf", "cifar_hf")]
    y = np.load(next((p for p in candidates if p.exists()), candidates[0]))["labels"]
    N = Z[0].shape[0]
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    fit, ev = idx[: N * 4 // 5], idx[N * 4 // 5:]

    Zz = [zc(z) for z in Z]
    aligned = [Zz[0]] + [procrustes_to(Zz[s], Zz[0]) for s in range(1, SEEDS)]
    stack = np.stack(aligned)
    rel = per_dim_rel(stack)

    reps = {
        "(a) single seed": Zz[0],
        "(b) naive mean (misaligned)": np.mean(Zz, 0),
        "(c) aligned mean": stack.mean(0),
        "(d) aligned + reliability-weighted": stack.mean(0) * rel,
    }
    print(f"per-dim reliability (aligned) median={np.median(rel):.3f}")
    print("=== downstream linear-probe accuracy (CIFAR-100 test split) ===")
    base = None
    for name, R in reps.items():
        acc = probe_acc(R[fit], y[fit], R[ev], y[ev])
        if base is None:
            base = acc
        print(f"{name:38s} acc={acc:.4f}  (vs single {acc-base:+.4f})")

if __name__ == "__main__":
    main()
