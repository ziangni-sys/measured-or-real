import argparse

import numpy as np
from scipy.stats import f as fdist
from scipy.stats import spearmanr

rng = np.random.default_rng(0)


def ucb_reliability(X, alpha=0.1):
    p, K = X.shape
    gm = X.mean()
    msb = K * ((X.mean(1) - gm) ** 2).sum() / (p - 1)
    msw = ((X - X.mean(1, keepdims=True)) ** 2).sum() / (p * (K - 1))
    Fobs = msb / msw
    d1, d2 = p - 1, p * (K - 1)
    Fu = Fobs / fdist.ppf(alpha, d1, d2)
    icc_u = (Fu - 1) / (Fu + K - 1)
    return icc_u


def run(N, K, rho1, beta, reps=4000, alpha=0.1):
    tau2 = 1.0
    sig2_base = tau2 * (1 - rho1) / rho1
    cover = 0
    spear = []
    for _ in range(reps):
        theta = rng.normal(0, np.sqrt(tau2), N)
        scale = np.exp(beta * (theta - theta.mean()))
        sig2_i = sig2_base * scale
        rho_pop = tau2 / (tau2 + sig2_i.mean())
        X = theta[:, None] + rng.normal(0, 1, (N, K)) * np.sqrt(sig2_i)[:, None]
        if _ < 200:
            spear.append(spearmanr(theta, X.var(1, ddof=1)).statistic)
        if ucb_reliability(X, alpha) >= rho_pop:
            cover += 1
    return cover / reps, float(np.mean(spear))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    print("UCB reliability upper-bound coverage of the population reliability (nominal 90%):")
    for beta, tag in [(0.0, "homoscedastic"), (0.3, "hetero (real-level)"), (0.6, "hetero (2x real)"), (1.2, "strong hetero")]:
        for rho1 in [0.5, 0.84]:
            cov, sp = run(N=200, K=4, rho1=rho1, beta=beta, reps=args.repetitions)
            print(f"  beta={beta} ({tag:20s}) rho={rho1}: coverage={cov:.3f}  [Spearman(mean,within-var)={sp:+.2f}]")
