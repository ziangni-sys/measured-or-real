import argparse

import numpy as np

rng = np.random.default_rng(0)


def icc_hat(N, K, rho_K, reps=20000, corrected=False):
    tau2 = 1.0
    sig2 = K * tau2 * (1 - rho_K) / rho_K
    est = np.empty(reps)
    for r in range(reps):
        mu = rng.normal(0, np.sqrt(tau2), N)[:, None]
        X = mu + rng.normal(0, np.sqrt(sig2), (N, K))
        gm = X.mean()
        msb = K * ((X.mean(1) - gm) ** 2).sum() / (N - 1)
        msw = ((X - X.mean(1, keepdims=True)) ** 2).sum() / (N * (K - 1))
        r_hat = 1 - msw / msb
        if corrected:
            r_hat = 1 - (1 - r_hat) * (N - 3) / (N - 1)
        est[r] = r_hat
    return est.mean(), est.std()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    print("E[rho_hat_K] vs exact 1-(1-rho_K)(N-1)/(N-3); bias is K-independent, O(1/N):")
    for N in [20, 50, 100, 1000]:
        for K in [2, 4, 20]:
            for rho in [0.5, 0.84]:
                m, sd = icc_hat(N, K, rho, reps=args.repetitions)
                pred = 1 - (1 - rho) * (N - 1) / (N - 3)
                mc, _ = icc_hat(N, K, rho, reps=args.repetitions, corrected=True)
                print(f"  N={N:5d} K={K:2d} rho={rho}: E[hat]={m:.4f} pred={pred:.4f} "
                      f"bias={m-rho:+.4f} sd={sd:.3f} | corrected E={mc:.4f} bias={mc-rho:+.4f}")
