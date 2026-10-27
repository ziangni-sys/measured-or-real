import argparse

import numpy as np
from scipy.stats import f as fdist
from scipy.stats import spearmanr

rng = np.random.default_rng(0)


def icc1_point(X):
    N, K = X.shape
    gm = X.mean()
    msb = K * ((X.mean(1) - gm) ** 2).sum() / (N - 1)
    msw = ((X - X.mean(1, keepdims=True)) ** 2).sum() / (N * (K - 1))
    return (msb - msw) / (msb + (K - 1) * msw) if (msb + (K - 1) * msw) > 0 else 0.0


def exact_f_ub(X, alpha):
    N, K = X.shape
    gm = X.mean()
    msb = K * ((X.mean(1) - gm) ** 2).sum() / (N - 1)
    msw = ((X - X.mean(1, keepdims=True)) ** 2).sum() / (N * (K - 1))
    Fu = (msb / msw) / fdist.ppf(alpha, N - 1, N * (K - 1))
    return (Fu - 1) / (Fu + K - 1)


def boot_ub(X, alpha, B=400):
    N = X.shape[0]
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, N, N)
        vals[b] = icc1_point(X[idx])
    return np.percentile(vals, 100 * (1 - alpha))


def run(N, K, rho1, beta, reps=1500, alpha=0.1, B=400):
    tau2 = 1.0
    sig2_base = tau2 * (1 - rho1) / rho1
    cov_f = cov_b = 0
    spear = []
    wid_f = wid_b = 0.0
    for r in range(reps):
        theta = rng.normal(0, np.sqrt(tau2), N)
        scale = np.exp(beta * (theta - theta.mean()))
        sig2_i = sig2_base * scale
        rho_pop = tau2 / (tau2 + sig2_i.mean())
        X = theta[:, None] + rng.normal(0, 1, (N, K)) * np.sqrt(sig2_i)[:, None]
        if r < 200:
            spear.append(spearmanr(theta, X.var(1, ddof=1)).statistic)
        uf = exact_f_ub(X, alpha)
        ub = boot_ub(X, alpha, B)
        cov_f += uf >= rho_pop
        cov_b += ub >= rho_pop
        wid_f += uf - icc1_point(X)
        wid_b += ub - icc1_point(X)
    return cov_f / reps, cov_b / reps, float(np.mean(spear)), wid_f / reps, wid_b / reps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=1500)
    parser.add_argument("--bootstrap-repetitions", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    print("Restoring UCB coverage under heteroscedasticity: exact-F vs unit-bootstrap upper bound")
    print("(single-seed reliability, nominal 90%, N=200, K=4)")
    print(f"{'beta':>5} {'Spearman':>9} {'rho1':>5} | {'cov(exactF)':>11} {'cov(boot)':>10} | {'width(F)':>8} {'width(boot)':>11}")
    for beta, _ in [(0.0, "homo"), (0.3, "real"), (0.6, "2x"), (1.2, "strong")]:
        for rho1 in (0.5, 0.84):
            cf, cb, sp, wf, wb = run(
                N=200,
                K=4,
                rho1=rho1,
                beta=beta,
                reps=args.repetitions,
                B=args.bootstrap_repetitions,
            )
            print(f"{beta:>5.1f} {sp:>+9.2f} {rho1:>5.2f} | {cf:>11.3f} {cb:>10.3f} | {wf:>8.3f} {wb:>11.3f}")
