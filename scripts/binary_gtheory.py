from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import betabinom

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    "Qwen2.5-0.5B-Instruct", "Qwen2-0.5B-Instruct", "Qwen2.5-1.5B-Instruct",
    "Qwen2-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct",
    "Qwen2-7B-Instruct",
    "SmolLM2-1.7B-Instruct", "SmolLM2-360M-Instruct", "SmolLM2-135M-Instruct",
    "Falcon3-1B-Instruct", "Falcon3-3B-Instruct", "Falcon3-7B-Instruct",
    "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct",
]


def load(tag):
    f = ROOT / "data" / f"llm_decomp_{tag}.npy"
    return np.load(f) if f.exists() else None


def gauss_oneway(X):
    N, K = X.shape
    gm = X.mean()
    msb = K * ((X.mean(1) - gm) ** 2).sum() / (N - 1)
    msw = ((X - X.mean(1, keepdims=True)) ** 2).sum() / (N * (K - 1))
    icc11 = (msb - msw) / (msb + (K - 1) * msw)
    v_i = max((msb - msw) / K, 0.0)
    g1 = v_i / (v_i + msw)
    return icc11, g1


def gauss_crossed(X):
    p, k = X.shape
    mi = X.mean(1, keepdims=True)
    mc = X.mean(0, keepdims=True)
    gm = X.mean()
    ms_i = (k * ((mi - gm) ** 2).sum()) / (p - 1)
    ms_res = ((X - mi - mc + gm) ** 2).sum() / ((p - 1) * (k - 1))
    v_i = max((ms_i - ms_res) / k, 0.0)
    return v_i / (v_i + ms_res)


def betabin_icc(counts, n):
    def nll(par):
        a, b = np.exp(par)
        return -betabinom.logpmf(counts, n, a, b).sum()
    m = counts.mean() / n
    v = counts.var() / n ** 2
    guess = max(m * (1 - m) / max(v, 1e-6) - 1, 0.5)
    x0 = np.log([max(m * guess, 0.1), max((1 - m) * guess, 0.1)])
    res = minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-4, maxiter=4000))
    a, b = np.exp(res.x)
    return 1.0 / (a + b + 1.0)


def logit_latent(counts, n):
    p = (counts + 0.5) / (n + 1.0)
    l = np.log(p / (1 - p))
    samp = 1.0 / (n * p * (1 - p))
    s2 = max(l.var(ddof=1) - samp.mean(), 0.0)
    return s2 / (s2 + np.pi ** 2 / 3.0)


if __name__ == "__main__":
    print("Binary item-difficulty reliability: Gaussian G-theory vs distribution-appropriate models")
    print(f"{'model':26s} {'acc':>5} {'g(1)cross':>9} {'ICC(1,1)':>9} {'beta-bin':>9} {'logit-lat':>9}")
    for tag in MODELS:
        C = load(tag)
        if C is None:
            continue
        X = C.reshape(-1, C.shape[-1]).astype(float).T
        counts = C.reshape(-1, C.shape[-1]).sum(0)
        n = X.shape[1]
        _, g1_ow = gauss_oneway(X)
        icc11, _ = gauss_oneway(X)
        g1_cx = gauss_crossed(X)
        bb = betabin_icc(counts, n)
        ll = logit_latent(counts, n)
        print(f"{tag:26s} {C.mean():>5.3f} {g1_cx:>9.3f} {icc11:>9.3f} {bb:>9.3f} {ll:>9.3f}")
