import argparse

import numpy as np
from scipy.stats import f as fdist, norm

RNG = np.random.default_rng(0)

def icc1k(M):
    k, n = M.shape
    mean = M.mean(0)
    msb = k * ((mean - mean.mean()) ** 2).sum() / (n - 1)
    msw = ((M - mean[None, :]) ** 2).sum() / (n * (k - 1))
    return (1 - msw / msb) if msb > 0 else 0.0, msb, msw

def reml_reliability(M):

    rho, msb, msw = icc1k(M)
    return max(rho, 0.0)

def pear(x, y):
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0

def var_f(d1, d2):
    if d2 <= 4:
        return np.nan
    return 2 * d2 * d2 * (d1 + d2 - 2) / (d1 * (d2 - 2) ** 2 * (d2 - 4))

def var_rho_hat(rho_hat, d1, d2):
    return (1 - rho_hat) ** 2 * var_f(d1, d2)

def fieller_ci(robs, gx, gy, n, k, alpha=0.1):

    z = norm.ppf(1 - alpha / 2)
    g = np.sqrt(max(gx * gy, 1e-12))
    v_rob = (1 - robs ** 2) ** 2 / (n - 3)
    d1, d2 = n * (k - 1), n - 1
    vgx, vgy = var_rho_hat(gx, d1, d2), var_rho_hat(gy, d1, d2)
    v_g = (g ** 2 / 4.0) * (vgx / max(gx, 1e-6) ** 2 + vgy / max(gy, 1e-6) ** 2)

    A = g * g - z * z * v_g
    B = -2 * g * robs
    C = robs * robs - z * z * v_rob
    disc = B * B - 4 * A * C
    if A <= 0 or disc < 0:
        return -1.0, 1.0
    s = np.sqrt(disc)
    lo, hi = (-B - s) / (2 * A), (-B + s) / (2 * A)
    return float(np.clip(min(lo, hi), -1, 1)), float(np.clip(max(lo, hi), -1, 1))

def bonferroni_ci(robs, gx, gy, n, k, alpha=0.1):

    a = alpha / 3
    d1, d2 = n * (k - 1), n - 1
    zr = np.arctanh(np.clip(robs, -0.999, 0.999))
    se = 1 / np.sqrt(n - 3)
    r_lo, r_hi = np.tanh(zr - norm.ppf(1 - a) * se), np.tanh(zr + norm.ppf(1 - a) * se)
    ql, qh = fdist.ppf(a, d1, d2), fdist.ppf(1 - a, d1, d2)
    def rel_ci(rh):
        return max(1 - (1 - rh) / ql, 1e-3), max(1 - (1 - rh) / qh, 1e-3)
    gx_hi, gx_lo = rel_ci(gx)
    gy_hi, gy_lo = rel_ci(gy)
    lo = r_lo / np.sqrt(max(gx_hi * gy_hi, 1e-9))
    hi = r_hi / np.sqrt(max(gx_lo * gy_lo, 1e-9))
    return float(np.clip(lo, -1, 1)), float(np.clip(hi, -1, 1))

def ucb_point(robs, gx, gy, n, k, alpha=0.1):
    d1, d2 = n * (k - 1), n - 1
    qg = fdist.ppf(1 - alpha, d2, d1)
    if 1 / max(1 - gx, 1e-9) <= qg or 1 / max(1 - gy, 1e-9) <= qg:
        return float(np.clip(robs, -1, 1))
    qu = fdist.ppf(1 - alpha, d1, d2)
    ux, uy = max(1 - (1 - gx) / qu, 1e-3), max(1 - (1 - gy) / qu, 1e-3)
    return float(np.clip(robs / np.sqrt(ux * uy), -1, 1))

def gen(n, k, rho, rstar):

    sig = np.sqrt(k * (1 - rho) / rho)
    z1 = RNG.standard_normal(n)
    z2 = RNG.standard_normal(n)
    xt, yt = z1, rstar * z1 + np.sqrt(1 - rstar ** 2) * z2
    X = xt[None, :] + sig * RNG.standard_normal((k, n))
    Y = yt[None, :] + sig * RNG.standard_normal((k, n))
    return X, Y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    global RNG
    RNG = np.random.default_rng(args.seed)
    n, alpha, reps = 50, 0.1, args.repetitions
    cells = [(0.2, 0.4), (0.3, 0.4), (0.2, 0.6), (0.5, 0.4), (0.2, 0.2)]

    X, Y = gen(n, 4, 0.3, 0.4)
    reml = reml_reliability(X[[0, 1]])
    pp = max(icc1k(X[[0, 1]])[0], 0.0)
    print(f"REML==positive-part check: reml={reml:.6f} pp={pp:.6f} equal={abs(reml-pp)<1e-12}\n")
    print("independent-design interval coverage of the true latent correlation (target=r*), N=50, K=2 split:")
    print("  rho   r*   Fieller(cov/width)   Bonferroni(cov/width)   UCB-pt|err|")
    for rho, rstar in cells:
        fc = fw = bc = bw = uerr = 0.0
        for _ in range(reps):

            X, Y = gen(n, 4, rho, rstar)
            gx, gy = max(icc1k(X[[0, 1]])[0], 1e-3), max(icc1k(Y[[0, 1]])[0], 1e-3)
            robs = pear(X[[2, 3]].mean(0), Y[[2, 3]].mean(0))
            flo, fhi = fieller_ci(robs, gx, gy, n, 2, alpha)
            blo, bhi = bonferroni_ci(robs, gx, gy, n, 2, alpha)
            fc += flo <= rstar <= fhi
            fw += fhi - flo
            bc += blo <= rstar <= bhi
            bw += bhi - blo
            uerr += abs(ucb_point(robs, gx, gy, n, 2, alpha) - rstar)
        r = reps
        print(f" {rho:.1f}  {rstar:.1f}   {fc/r:.3f} / {fw/r:.3f}      "
              f"{bc/r:.3f} / {bw/r:.3f}       {uerr/r:.3f}")

if __name__ == "__main__":
    main()
