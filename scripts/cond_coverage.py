import argparse

import numpy as np
from scipy.stats import f as fdist

rng = np.random.default_rng(0)


def icc_k(X):
    N, K = X.shape
    gm = X.mean()
    msb = K * ((X.mean(1) - gm) ** 2).sum() / (N - 1)
    msw = ((X - X.mean(1, keepdims=True)) ** 2).sum() / (N * (K - 1))
    return 1.0 - msw / msb if msb > 0 else -np.inf


def run(N, K, rho_k, rstar, reps=20000, alpha=0.1):
    d1, d2 = N * (K - 1), N - 1
    q_gate = fdist.ppf(1 - alpha, d2, d1)
    q_up = fdist.ppf(1 - alpha, d1, d2)
    rho1 = rho_k / (K - (K - 1) * rho_k)
    sig2 = (1 - rho1) / rho1
    cov = np.zeros(reps, bool)
    passed = np.zeros(reps, bool)
    rhat = np.zeros(reps)
    for r in range(reps):
        L = rng.multivariate_normal([0, 0], [[1, rstar], [rstar, 1]], N)
        tx, ty = L[:, 0], L[:, 1]
        XA = tx[:, None] + rng.normal(0, np.sqrt(sig2), (N, K))
        YA = ty[:, None] + rng.normal(0, np.sqrt(sig2), (N, K))
        XB = tx[:, None] + rng.normal(0, np.sqrt(sig2), (N, K))
        YB = ty[:, None] + rng.normal(0, np.sqrt(sig2), (N, K))
        rhx, rhy = icc_k(XA), icc_k(YA)
        robs = np.corrcoef(XB.mean(1), YB.mean(1))[0, 1]
        ux = max(1 - (1 - rhx) / q_up, 1e-3)
        uy = max(1 - (1 - rhy) / q_up, 1e-3)
        cov[r] = rho_k <= ux
        gx = 1.0 / max(1 - rhx, 1e-9) > q_gate
        gy = 1.0 / max(1 - rhy, 1e-9) > q_gate
        passed[r] = gx and gy
        rhat[r] = np.clip(robs / np.sqrt(ux * uy), -1, 1) if passed[r] else np.clip(robs, -1, 1)
    pr = passed.mean()
    marg = cov.mean()
    cond = cov[passed].mean() if passed.any() else np.nan
    bias_p = (rhat[passed] - rstar).mean() if passed.any() else np.nan
    mse_p = ((rhat[passed] - rstar) ** 2).mean() if passed.any() else np.nan
    mse_all = ((rhat - rstar) ** 2).mean()
    return dict(pass_rate=pr, cov_marg=marg, cov_cond=cond, bias_pass=bias_p, mse_pass=mse_p, mse_all=mse_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    print("Conditional coverage/bias/MSE given the gate passes (Alg 1; nominal UCB 90%, alpha=0.1)")
    print(f"{'N':>4} {'K':>3} {'rho_K':>6} {'r*':>5} | {'pass%':>6} {'cov(marg)':>9} {'cov|pass':>9} {'bias|pass':>9} {'MSE|pass':>8} {'MSE(all)':>8}")
    for N in (50, 200):
        for rho_k in (0.3, 0.6, 0.9):
            for rstar in (0.0, 0.4):
                d = run(N, 4, rho_k, rstar, reps=args.repetitions)
                print(f"{N:>4} {4:>3} {rho_k:>6.2f} {rstar:>5.2f} | {100*d['pass_rate']:>6.1f} "
                      f"{d['cov_marg']:>9.3f} {d['cov_cond']:>9.3f} {d['bias_pass']:>+9.3f} {d['mse_pass']:>8.4f} {d['mse_all']:>8.4f}")
