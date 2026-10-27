import argparse

import numpy as np

RNG = np.random.default_rng(0)

def icc1k(M):
    k, n = M.shape
    mean = M.mean(0)
    msb = k * ((mean - mean.mean()) ** 2).sum() / (n - 1)
    msw = ((M - mean[None, :]) ** 2).sum() / (n * (k - 1))
    return (1 - msw / msb) if msb > 0 else 0.0

def pear(x, y):
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0

def sample(n, k, rho, rstar, theta=None):
    sig = np.sqrt(k * (1 - rho) / rho)
    if theta is None:
        z1, z2 = RNG.standard_normal(n), RNG.standard_normal(n)
        tx, ty = z1, rstar * z1 + np.sqrt(1 - rstar ** 2) * z2
    else:
        tx, ty = theta
    X = tx[None, :] + sig * RNG.standard_normal((k, n))
    Y = ty[None, :] + sig * RNG.standard_normal((k, n))
    return X, Y, (tx, ty)

def stats(X, Y):
    rx = icc1k(X[[0, 1]]); ry = icc1k(Y[[0, 1]])
    robs = pear(X[[2, 3]].mean(0), Y[[2, 3]].mean(0))
    return robs, rx, ry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=20000)
    parser.add_argument("--large-repetitions", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    global RNG
    RNG = np.random.default_rng(args.seed)
    n, k, rho, rstar, reps = 200, 4, 0.3, 0.4, args.repetitions

    ro, rx = [], []
    for _ in range(reps):
        X, Y, _ = sample(n, k, rho, rstar)
        s = stats(X, Y); ro.append(s[0]); rx.append(s[1])
    print(f"(a) MARGINAL corr(r_obs, rho_hat_X) [fresh latent each rep] = {pear(np.array(ro), np.array(rx)):+.4f}")
    print("    -> tiny (within MC noise of 0): the marginal dependence is negligible in magnitude,")
    print("       but NOT guaranteed exactly 0 (they share the latent), so we do not rely on marginal independence.")

    _, _, theta = sample(n, k, rho, rstar)
    ro, rx = [], []
    for _ in range(reps):
        X, Y, _ = sample(n, k, rho, rstar, theta=theta)
        s = stats(X, Y); ro.append(s[0]); rx.append(s[1])
    print(f"(b) CONDITIONAL corr(r_obs, rho_hat_X) | latent fixed = {pear(np.array(ro), np.array(rx)):+.4f}")
    print("    -> ~0 confirms conditional independence given theta* (A-noise _|_ B-noise). This is what the proof needs.")

    print("(c) E[r_hat^2] and E[1/rho_hat_X] under the independent design (guard eps):")
    big = args.large_repetitions
    ro = np.empty(big); rxa = np.empty(big); rya = np.empty(big)
    for i in range(big):
        X, Y, _ = sample(n, k, rho, rstar)
        s = stats(X, Y); ro[i], rxa[i], rya[i] = s
    for eps in [1e-2, 1e-3, 1e-4, 1e-6]:
        gx = np.maximum(rxa, eps); gy = np.maximum(rya, eps)
        m2 = np.mean((ro ** 2) / (gx * gy))
        inv = np.mean(1.0 / gx)
        print(f"    eps={eps:.0e}:  E[r_hat^2]={m2:10.1f}   E[1/rho_hat_X]={inv:8.2f}   P(rho_hat_X<=0)={np.mean(rxa<=0):.3f}")
    print("    -> growth without bound as eps->0 confirms the infinite second moment (conditional argument).")

if __name__ == "__main__":
    main()
