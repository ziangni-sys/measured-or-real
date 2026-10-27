import os
from pathlib import Path

import numpy as np
from scipy.stats import f as fdist
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATASET = os.environ.get("DATASET", "cifar100")
SDIR = ROOT / "data" / ("prune_scores" if DATASET == "cifar100" else f"prune_scores_{DATASET}")


def seed_subset_spread(X, sub, nboot=300):
    p, r = X.shape
    rng = np.random.default_rng(0)
    vals = [two_way(X[:, rng.choice(r, sub, replace=False)])["icc1"] for _ in range(nboot)]
    return float(np.mean(vals)), float(np.std(vals)), np.percentile(vals, [2.5, 97.5])

def two_way(X):

    p, r = X.shape
    g = X.mean()
    mp = X.mean(1)
    mr = X.mean(0)
    ms_p = r * ((mp - g) ** 2).sum() / (p - 1)
    ms_r = p * ((mr - g) ** 2).sum() / (r - 1)
    resid = X - mp[:, None] - mr[None, :] + g
    ms_e = (resid ** 2).sum() / ((p - 1) * (r - 1))
    v_p = max((ms_p - ms_e) / r, 0.0)
    v_r = max((ms_r - ms_e) / p, 0.0)
    v_e = ms_e

    ms_w = (ms_r * (r - 1) + ms_e * (p - 1) * (r - 1)) / (p * (r - 1))
    icc1 = (ms_p - ms_w) / (ms_p + (r - 1) * ms_w)

    icc_abs = v_p / (v_p + v_r + v_e)
    icc_con = v_p / (v_p + v_e)

    Fs = ms_p / ms_e
    d1, d2 = p - 1, (p - 1) * (r - 1)
    fl = Fs / fdist.ppf(0.975, d1, d2)
    fu = Fs / fdist.ppf(0.025, d1, d2)
    lo = (fl - 1) / (fl + r - 1)
    hi = (fu - 1) / (fu + r - 1)
    return dict(v_p=v_p, v_seed=v_r, v_resid=v_e, icc1=icc1, icc_abs=icc_abs,
                icc_con=icc_con, con_lo=lo, con_hi=hi, seed_share=v_r / (v_p + v_r + v_e))

def main():
    files = sorted(SDIR.glob("seed*_e30.npz"), key=lambda p: int(p.stem.split("_")[0][4:]))
    K = len(files)
    for name in ["el", "du"]:
        X = np.stack([np.load(f)[name] for f in files]).T
        r = two_way(X)
        lbl = "EL2N" if name == "el" else "Dyn-Unc"
        print(f"=== {lbl} ({DATASET}): example x seed two-way G-theory (p={X.shape[0]}, r={X.shape[1]} seeds) ===")
        print(f"  variance: example={r['v_p']:.4f}  seed-main={r['v_seed']:.5f}  residual={r['v_resid']:.4f}"
              f"  (seed share {r['seed_share']*100:.2f}%)")
        print(f"  single-seed reliability: one-way ICC(1,1)={r['icc1']:.3f}  "
              f"two-way consistency ICC(C,1)={r['icc_con']:.3f} [95%CI {r['con_lo']:.3f},{r['con_hi']:.3f}]  "
              f"absolute ICC(2,1)={r['icc_abs']:.3f}")
        if K >= 6:
            sub = K // 2
            m, sd, ci = seed_subset_spread(X, sub)
            print(f"  seed-population sensitivity: ICC(1,1) over random {sub}-seed subsets of {K}"
                  f" = {m:.3f} +-{sd:.3f} [95% {ci[0]:.3f},{ci[1]:.3f}]")
        mu, wv = X.mean(1), X.var(1, ddof=1)
        rho = spearmanr(mu, wv).statistic
        q = np.quantile(mu, [0, .25, .5, .75, 1.0])
        sds = [np.sqrt(wv[(mu >= q[i]) & (mu <= q[i + 1])]).mean() for i in range(4)]
        print(f"  homoscedasticity: Spearman(mean, within-var)={rho:+.3f}  "
              f"within-example SD by mean-quartile {'/'.join(f'{s:.3f}' for s in sds)}")

if __name__ == "__main__":
    main()
