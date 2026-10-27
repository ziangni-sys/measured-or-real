from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FDIR = ROOT / "data" / "gauge_feats"
EPOCHS = 20
SEEDS = 3

def zc(Z):
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-8)

def per_dim_rel(Za, Zb):
    a, b = zc(Za), zc(Zb)
    num = (a * b).sum(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0)) + 1e-9
    return float(np.mean(num / den))

def procrustes_R(Zs, Z0):
    u, _, vt = np.linalg.svd(Zs.T @ Z0, full_matrices=False)
    return u @ vt

def dist_rel(Za, Zb, rng, npair=30000):
    n = Za.shape[0]
    i, j = rng.integers(0, n, npair), rng.integers(0, n, npair)
    da = np.linalg.norm(Za[i] - Za[j], axis=1)
    db = np.linalg.norm(Zb[i] - Zb[j], axis=1)
    return float(np.corrcoef(da, db)[0, 1])

def cka(Za, Zb):
    a, b = Za - Za.mean(0), Zb - Zb.mean(0)
    hsic = np.linalg.norm(a.T @ b) ** 2
    return float(hsic / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b) + 1e-12))

def main():
    if not all((FDIR / f"feat_seed{s}_e{EPOCHS}.npy").exists() for s in range(SEEDS)):
        raise SystemExit("missing data/gauge_feats/feat_seed{0,1,2}_e20.npy; see KNOWN_LIMITATIONS.md")
    Z = [np.load(FDIR / f"feat_seed{s}_e{EPOCHS}.npy") for s in range(SEEDS)]
    n = Z[0].shape[0]
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    fit, ev = idx[: n // 2], idx[n // 2:]

    raw, proc, dr, ck = [], [], [], []
    for a in range(SEEDS):
        for b in range(a + 1, SEEDS):
            Za, Zb = zc(Z[a]), zc(Z[b])
            raw.append(per_dim_rel(Za[ev], Zb[ev]))

            R = procrustes_R(Za[fit], Zb[fit])
            proc.append(per_dim_rel((Za @ R)[ev], Zb[ev]))
            dr.append(dist_rel(Z[a][ev], Z[b][ev], rng))
            ck.append(cka(Z[a][ev], Z[b][ev]))
    print("=== GAUGE PROBE (held-out eval; Procrustes fit on disjoint split) ===")
    print(f"(1) raw per-dim reliability            : {np.mean(raw):.3f}")
    print(f"(2) Procrustes-aligned per-dim (held-out): {np.mean(proc):.3f}")
    print(f"(3a) rotation-invariant dist-rel        : {np.mean(dr):.3f}")
    print(f"(3b) linear CKA                         : {np.mean(ck):.3f}")

if __name__ == "__main__":
    main()
