from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f as f_dist


matplotlib.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "axes.linewidth": 0.6})
ROOT = Path(__file__).resolve().parents[1]

N = 50
K = 3
R_TRUE = 0.4
ALPHA = 0.1
RHO_GRID = [0.1, 0.2, 0.4, 0.6, 0.8]
DENOM_EPS = 1e-12

NOISE_LABELS = {
    "gaussian": "Gaussian",
    "student_t3": "Student-t(3)",
    "contaminated": "95/5 contaminated Gaussian",
}

METHOD_ORDER = [
    "naive_plugin",
    "clipped_plugin",
    "fixed_floor_0.1",
    "bootstrap_ucb",
    "clipping_only",
    "gate_point",
    "ucb_no_gate",
    "algorithm1",
]

METHOD_LABELS = {
    "naive_plugin": "naive plug-in",
    "clipped_plugin": "clipped plug-in",
    "fixed_floor_0.1": "fixed floor c=0.1",
    "bootstrap_ucb": "Bootstrap-UCB",
    "clipping_only": "clipping only",
    "gate_point": "gate + point estimate",
    "ucb_no_gate": "UCB without gate",
    "algorithm1": "Algorithm 1",
}


def clip(v: float) -> float:
    return float(np.clip(v, -1.0, 1.0))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    den = math.sqrt(float(np.dot(xc, xc) * np.dot(yc, yc)))
    if den == 0.0:
        return 0.0
    return float(np.dot(xc, yc) / den)


def icc_1k(rep: np.ndarray) -> tuple[float, np.ndarray]:
    n, k = rep.shape
    mean = rep.mean(axis=1)
    grand = float(mean.mean())
    ssb = float(np.square(mean - grand).sum())
    msb = ssb * k / (n - 1.0)
    ssw = float(np.square(rep - mean[:, None]).sum())
    msw = ssw / (n * (k - 1.0))
    rho = 0.0 if msb <= 0.0 else 1.0 - msw / msb
    return float(rho), mean


def icc_1k_batch(samples: np.ndarray) -> np.ndarray:
    # samples shape: bootstrap x N x K
    b, n, k = samples.shape
    mean = samples.mean(axis=2)
    grand = mean.mean(axis=1)
    ssb = np.square(mean - grand[:, None]).sum(axis=1)
    msb = ssb * k / (n - 1.0)
    ssw = np.square(samples - mean[:, :, None]).sum(axis=(1, 2))
    msw = ssw / (n * (k - 1.0))
    return np.where(msb <= 0.0, 0.0, 1.0 - msw / msb)


def noise_matrix(rng: np.random.Generator, noise: str, sigma: float, shape: tuple[int, int]) -> np.ndarray:
    if noise == "gaussian":
        return rng.normal(0.0, sigma, size=shape)
    if noise == "student_t3":
        return sigma / math.sqrt(3.0) * rng.standard_t(df=3, size=shape)
    if noise == "contaminated":
        sigma_base = sigma / math.sqrt(2.2)
        large = rng.random(size=shape) < 0.05
        sd = np.where(large, 5.0 * sigma_base, sigma_base)
        return rng.normal(0.0, sd)
    raise ValueError(f"unknown noise type: {noise}")


def simulate_dataset(rng: np.random.Generator, rho: float, noise: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    sigma = math.sqrt(K * (1.0 - rho) / rho)
    z1 = rng.normal(size=N)
    z2 = rng.normal(size=N)
    x_latent = z1
    y_latent = R_TRUE * z1 + math.sqrt(1.0 - R_TRUE**2) * z2
    x = x_latent[:, None] + noise_matrix(rng, noise, sigma, (N, K))
    y = y_latent[:, None] + noise_matrix(rng, noise, sigma, (N, K))
    return x, y, {"sigma2": sigma**2}


@dataclass
class Est:
    corr_obs: float
    rho_hat_x: float
    rho_hat_y: float
    x: np.ndarray
    y: np.ndarray


@dataclass
class EvalResult:
    value: float
    raw: float
    clipped: bool = False
    refused: bool = False
    undefined_raw: bool = False
    invalid_ucb: bool = False
    bootstrap_failure: bool = False
    conservative_event: bool | None = None


@dataclass
class Accumulator:
    values: list[float] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    abs_errors: list[float] = field(default_factory=list)
    raw_over_one: int = 0
    clipped: int = 0
    refused: int = 0
    undefined_raw: int = 0
    invalid_ucb: int = 0
    bootstrap_failure: int = 0
    conservative_success: int = 0
    conservative_total: int = 0
    runtime: float = 0.0

    def add(self, result: EvalResult, elapsed: float) -> None:
        if not math.isfinite(result.value):
            raise AssertionError("method returned a non-finite value")
        err = result.value - R_TRUE
        self.values.append(result.value)
        self.errors.append(err)
        self.abs_errors.append(abs(err))
        self.runtime += elapsed
        if math.isfinite(result.raw) and abs(result.raw) > 1.0:
            self.raw_over_one += 1
        if result.clipped:
            self.clipped += 1
        if result.refused:
            self.refused += 1
        if result.undefined_raw:
            self.undefined_raw += 1
        if result.invalid_ucb:
            self.invalid_ucb += 1
        if result.bootstrap_failure:
            self.bootstrap_failure += 1
        if result.conservative_event is not None:
            self.conservative_total += 1
            self.conservative_success += int(result.conservative_event)

    def row(self, noise: str, rho: float, method: str, reps: int) -> dict[str, object]:
        vals = np.asarray(self.values, dtype=float)
        errs = np.asarray(self.errors, dtype=float)
        abs_errs = np.asarray(self.abs_errors, dtype=float)
        if vals.size != reps:
            raise AssertionError(f"missing outputs for {noise}/{rho}/{method}: {vals.size} != {reps}")
        mse = float(np.mean(np.square(errs)))
        if not math.isfinite(mse):
            raise AssertionError(f"non-finite MSE for {noise}/{rho}/{method}")
        conservative = ""
        if self.conservative_total:
            conservative = self.conservative_success / self.conservative_total
        return {
            "noise": noise,
            "noise_label": NOISE_LABELS[noise],
            "rho": rho,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "mse": mse,
            "bias": float(np.mean(errs)),
            "variance": float(np.var(vals, ddof=1)) if vals.size > 1 else 0.0,
            "median_abs_error": float(np.quantile(abs_errs, 0.50)),
            "p90_abs_error": float(np.quantile(abs_errs, 0.90)),
            "p99_abs_error": float(np.quantile(abs_errs, 0.99)),
            "raw_over_one_rate": self.raw_over_one / reps,
            "clipping_rate": self.clipped / reps,
            "refusal_or_gate_rate": self.refused / reps,
            "undefined_or_nonfinite_raw_rate": self.undefined_raw / reps,
            "mean_runtime_seconds": self.runtime / reps,
            "invalid_ucb_rate": self.invalid_ucb / reps,
            "bootstrap_failure_or_degenerate_rate": self.bootstrap_failure / reps,
            "alg1_conservative_frequency": conservative,
        }


def estimate(x: np.ndarray, y: np.ndarray) -> Est:
    rx, xbar = icc_1k(x)
    ry, ybar = icc_1k(y)
    return Est(corr_obs=pearson(xbar, ybar), rho_hat_x=rx, rho_hat_y=ry, x=x, y=y)


def f_quantile(p: float, d1: float, d2: float) -> float:
    return float(f_dist.ppf(p, d1, d2))


def gate_passes(rho_hat: float, alpha: float) -> bool:
    d1 = N * (K - 1)
    d2 = N - 1
    q_gate = f_quantile(1.0 - alpha, d2, d1)
    return 1.0 / max(1.0 - rho_hat, 1e-9) > q_gate


def reliability_ucb(rho_hat: float, alpha: float) -> tuple[float, bool]:
    d1 = N * (K - 1)
    d2 = N - 1
    q_up = f_quantile(1.0 - alpha, d1, d2)
    raw_u = 1.0 - (1.0 - rho_hat) / q_up
    invalid = (not math.isfinite(raw_u)) or raw_u <= 0.0
    return max(raw_u, 1e-3), invalid


def naive_plugin(est: Est) -> EvalResult:
    prod = est.rho_hat_x * est.rho_hat_y
    raw_unhandled = est.corr_obs / math.sqrt(prod) if prod > 0.0 else math.nan
    raw_guarded = est.corr_obs / math.sqrt(max(prod, 1e-9))
    return EvalResult(value=raw_guarded, raw=raw_guarded, undefined_raw=not math.isfinite(raw_unhandled))


def clipped_plugin(est: Est) -> EvalResult:
    base = naive_plugin(est)
    raw = base.raw
    val = clip(raw)
    return EvalResult(value=val, raw=raw, clipped=abs(val - raw) > 1e-12, undefined_raw=base.undefined_raw)


def fixed_floor(est: Est, floor: float = 0.1) -> EvalResult:
    raw = est.corr_obs / math.sqrt(max(est.rho_hat_x, floor) * max(est.rho_hat_y, floor))
    val = clip(raw)
    return EvalResult(value=val, raw=raw, clipped=abs(val - raw) > 1e-12)


def ucb_no_gate(est: Est, alpha: float) -> EvalResult:
    ux, bad_x = reliability_ucb(est.rho_hat_x, alpha)
    uy, bad_y = reliability_ucb(est.rho_hat_y, alpha)
    raw = est.corr_obs / math.sqrt(ux * uy)
    val = clip(raw)
    return EvalResult(value=val, raw=raw, clipped=abs(val - raw) > 1e-12, invalid_ucb=bad_x or bad_y)


def gate_point(est: Est, alpha: float) -> EvalResult:
    if not gate_passes(est.rho_hat_x, alpha) or not gate_passes(est.rho_hat_y, alpha):
        raw = est.corr_obs
        return EvalResult(value=clip(raw), raw=raw, refused=True)
    base = naive_plugin(est)
    raw = base.raw
    val = clip(raw)
    return EvalResult(value=val, raw=raw, clipped=abs(val - raw) > 1e-12, undefined_raw=base.undefined_raw)


def algorithm1(est: Est, rho: float, alpha: float) -> EvalResult:
    if not gate_passes(est.rho_hat_x, alpha) or not gate_passes(est.rho_hat_y, alpha):
        raw = est.corr_obs
        val = clip(raw)
        oracle = abs(est.corr_obs / math.sqrt(rho * rho))
        return EvalResult(value=val, raw=raw, refused=True, conservative_event=abs(val) <= oracle + 1e-12)
    ux, bad_x = reliability_ucb(est.rho_hat_x, alpha)
    uy, bad_y = reliability_ucb(est.rho_hat_y, alpha)
    raw = est.corr_obs / math.sqrt(ux * uy)
    val = clip(raw)
    oracle = abs(est.corr_obs / math.sqrt(rho * rho))
    return EvalResult(
        value=val,
        raw=raw,
        clipped=abs(val - raw) > 1e-12,
        invalid_ucb=bad_x or bad_y,
        conservative_event=abs(val) <= oracle + 1e-12,
    )


def bootstrap_ucb(est: Est, rng: np.random.Generator, alpha: float, b_reps: int) -> EvalResult:
    idx = rng.integers(0, N, size=(b_reps, N))
    bx = icc_1k_batch(est.x[idx, :])
    by = icc_1k_batch(est.y[idx, :])
    qx = np.quantile(bx, [alpha, 1.0 - alpha])
    qy = np.quantile(by, [alpha, 1.0 - alpha])
    nonfinite = (not np.all(np.isfinite(qx))) or (not np.all(np.isfinite(qy)))
    degenerate = (
        (not nonfinite)
        and (qx[1] <= qx[0] + 1e-12 or qy[1] <= qy[0] + 1e-12)
    )
    failure = (
        nonfinite
        or degenerate
    )
    if nonfinite or qx[0] <= 0.0 or qy[0] <= 0.0:
        raw = est.corr_obs
        return EvalResult(value=clip(raw), raw=raw, refused=True, bootstrap_failure=failure)
    ux = max(float(qx[1]), DENOM_EPS)
    uy = max(float(qy[1]), DENOM_EPS)
    raw = est.corr_obs / math.sqrt(ux * uy)
    val = clip(raw)
    return EvalResult(value=val, raw=raw, clipped=abs(val - raw) > 1e-12, bootstrap_failure=failure)


def evaluate_methods(est: Est, rho: float, rng: np.random.Generator, alpha: float, b_reps: int) -> dict[str, EvalResult]:
    return {
        "naive_plugin": naive_plugin(est),
        "clipped_plugin": clipped_plugin(est),
        "fixed_floor_0.1": fixed_floor(est, 0.1),
        "bootstrap_ucb": bootstrap_ucb(est, rng, alpha, b_reps),
        "clipping_only": clipped_plugin(est),
        "gate_point": gate_point(est, alpha),
        "ucb_no_gate": ucb_no_gate(est, alpha),
        "algorithm1": algorithm1(est, rho, alpha),
    }


def run_noise_diagnostics(rng: np.random.Generator, noise_types: list[str]) -> list[dict[str, float]]:
    rows = []
    for rho in RHO_GRID:
        sigma2 = K * (1.0 - rho) / rho
        sigma = math.sqrt(sigma2)
        for noise in noise_types:
            e = noise_matrix(rng, noise, sigma, (250_000,))
            empirical = float(np.var(e, ddof=1))
            rel_error = abs(empirical - sigma2) / sigma2
            tolerance = 0.25 if noise == "student_t3" else 0.08
            if rel_error > tolerance:
                raise AssertionError(
                    f"{noise} variance diagnostic failed at rho={rho}: "
                    f"empirical={empirical:.4g}, target={sigma2:.4g}, rel_error={rel_error:.3f}"
                )
            rows.append(
                {
                    "noise": noise,
                    "rho": rho,
                    "target_variance": sigma2,
                    "empirical_variance": empirical,
                    "relative_error": rel_error,
                }
            )
    return rows


def run_gaussian_consistency_diagnostic(rng: np.random.Generator) -> dict[str, float]:
    rho = 0.2
    reps = 600
    rx = []
    mse_alg1 = []
    mse_clip = []
    for _ in range(reps):
        x, y, _ = simulate_dataset(rng, rho, "gaussian")
        est = estimate(x, y)
        rx.append(est.rho_hat_x)
        mse_alg1.append((algorithm1(est, rho, ALPHA).value - R_TRUE) ** 2)
        mse_clip.append((clipped_plugin(est).value - R_TRUE) ** 2)
    expected_bias = -2.0 * (1.0 - rho) / (N - 3.0)
    mean_rho_hat = float(np.mean(rx))
    expected_mean = rho + expected_bias
    if abs(mean_rho_hat - expected_mean) > 0.08:
        raise AssertionError(
            f"Gaussian consistency diagnostic failed: mean rho_hat={mean_rho_hat:.3f}, "
            f"expected near {expected_mean:.3f}"
        )
    return {
        "rho": rho,
        "mean_rho_hat_x": mean_rho_hat,
        "expected_mean_rho_hat_x": expected_mean,
        "mse_algorithm1": float(np.mean(mse_alg1)),
        "mse_clipped_plugin": float(np.mean(mse_clip)),
    }


def make_main_figure(rows: list[dict[str, object]], out: Path) -> None:
    colors = {
        "clipped_plugin": "#E67E22",
        "fixed_floor_0.1": "#7F8C8D",
        "bootstrap_ucb": "#27AE60",
        "algorithm1": "#2B7CCE",
    }
    markers = {
        "clipped_plugin": "s-",
        "fixed_floor_0.1": "d--",
        "bootstrap_ucb": "o-",
        "algorithm1": "^-",
    }
    methods = ["clipped_plugin", "fixed_floor_0.1", "bootstrap_ucb", "algorithm1"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), sharey=True)
    for ax, noise in zip(axes, ["gaussian", "student_t3", "contaminated"], strict=True):
        for method in methods:
            rs = sorted((r for r in rows if r["noise"] == noise and r["method"] == method), key=lambda x: x["rho"])
            ax.plot(
                [r["rho"] for r in rs],
                [r["mse"] for r in rs],
                markers[method],
                color=colors[method],
                lw=1.6 if method == "algorithm1" else 1.3,
                ms=3.8,
                label=METHOD_LABELS[method],
            )
        ax.set_title(NOISE_LABELS[noise], fontsize=9)
        ax.set_xlabel("reliability $\\rho$")
        ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("MSE vs. true correlation")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def make_ablation_table(rows: list[dict[str, object]], out: Path) -> None:
    ablation_methods = ["clipping_only", "gate_point", "ucb_no_gate", "algorithm1"]
    focus_rhos = [0.1, 0.2, 0.4]
    with out.open("w", encoding="utf-8") as f:
        f.write("| noise | rho | clipping only | gate + point | UCB no gate | Algorithm 1 |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for noise in ["gaussian", "student_t3", "contaminated"]:
            for rho in focus_rhos:
                vals = {}
                for method in ablation_methods:
                    row = next(r for r in rows if r["noise"] == noise and r["rho"] == rho and r["method"] == method)
                    vals[method] = float(row["mse"])
                f.write(
                    f"| {NOISE_LABELS[noise]} | {rho:.1f} | "
                    f"{vals['clipping_only']:.4g} | {vals['gate_point']:.4g} | "
                    f"{vals['ucb_no_gate']:.4g} | {vals['algorithm1']:.4g} |\n"
                )


def write_report(
    rows: list[dict[str, object]],
    config: dict[str, object],
    diagnostics: dict[str, object],
    output_dir: Path,
    elapsed: float,
) -> None:
    def get(noise: str, rho: float, method: str, key: str = "mse") -> float:
        row = next(r for r in rows if r["noise"] == noise and r["rho"] == rho and r["method"] == method)
        return float(row[key])

    alg1_gauss_cons = [
        float(r["alg1_conservative_frequency"])
        for r in rows
        if r["noise"] == "gaussian" and r["method"] == "algorithm1" and r["alg1_conservative_frequency"] != ""
    ]
    bootstrap_slower = np.mean(
        [
            get(noise, rho, "bootstrap_ucb", "mean_runtime_seconds")
            / max(get(noise, rho, "algorithm1", "mean_runtime_seconds"), 1e-12)
            for noise in config["noise_types"]
            for rho in config["rho_grid"]
        ]
    )
    wins = []
    for noise in config["noise_types"]:
        for rho in config["rho_grid"]:
            deployable = ["clipped_plugin", "fixed_floor_0.1", "bootstrap_ucb", "algorithm1"]
            mses = {m: get(noise, rho, m) for m in deployable}
            best = min(mses, key=mses.get)
            if best != "algorithm1" and mses[best] < mses["algorithm1"] * 0.9:
                wins.append((noise, rho, best, mses[best], mses["algorithm1"]))

    try:
        shown_dir = output_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown_dir = output_dir.name
    cargo_available = shutil.which("cargo") is not None

    lines = [
        "# Disattenuation Robustness Experiment",
        "",
        "## Commands",
        "",
        f"- This run: `uv run python scripts/disatt_robustness.py --repetitions {config['repetitions']} --bootstrap-repetitions {config['bootstrap_repetitions']} --seed {config['seed']} --output-dir {shown_dir}`.",
        "- Rust simulation of the same rule: `cargo run --release -p erel-cli -- disatt`.",
        "",
        "## Runtime and hardware",
        "",
        f"- Total wall time: {elapsed:.1f} seconds.",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        f"- Processor: {platform.processor() or 'not reported'}",
        "",
        "## Implementation checks",
        "",
        "- Algorithm 1 in this script uses the same gate, Exact-F quantiles, UCB formula, conservative denominator, and clipping as `crates/erel-core/src/disatt.rs`.",
        "- The target reliability is the K-replicate mean reliability with `sigma^2 = K * (1 - rho) / rho` and latent variance 1.",
        f"- Gaussian diagnostic at rho={diagnostics['gaussian_consistency']['rho']}: mean ICC {diagnostics['gaussian_consistency']['mean_rho_hat_x']:.3f}, expected finite-sample mean {diagnostics['gaussian_consistency']['expected_mean_rho_hat_x']:.3f}.",
        "- Noise variance diagnostics are in `noise_variance_diagnostics.csv`; each condition matched the target marginal variance within the scripted tolerance.",
        "",
        "## Main quantitative findings",
        "",
    ]
    for noise in config["noise_types"]:
        m_alg_low = get(noise, 0.1, "algorithm1")
        m_boot_low = get(noise, 0.1, "bootstrap_ucb")
        m_alg_mid = get(noise, 0.4, "algorithm1")
        m_floor_mid = get(noise, 0.4, "fixed_floor_0.1")
        lines.append(
            f"- {NOISE_LABELS[noise]}: at rho=0.1 Algorithm 1 MSE={m_alg_low:.4g}, Bootstrap-UCB MSE={m_boot_low:.4g}; at rho=0.4 Algorithm 1 MSE={m_alg_mid:.4g}, fixed floor MSE={m_floor_mid:.4g}."
        )
    lines.append(
        f"- Mean Bootstrap-UCB runtime was {bootstrap_slower:.1f}x Algorithm 1 runtime per dataset across cells."
    )
    if alg1_gauss_cons:
        lines.append(
            f"- Gaussian Algorithm 1 conservative-frequency range: {min(alg1_gauss_cons):.3f} to {max(alg1_gauss_cons):.3f}; nominal lower bound is {1 - 2 * ALPHA:.1f}."
        )
    lines += [
        "- The component ablation (clipping only, gate + point estimate, UCB without gate, full Algorithm 1) is in `ablation_mse.md`.",
        "- The Exact-F conservative-frequency guarantee holds for Gaussian noise only; the Student-t and contaminated cells are empirical.",
        "",
        "## Algorithm 1 losses of at least 10% MSE",
        "",
    ]
    if wins:
        for noise, rho, best, best_mse, alg_mse in wins:
            lines.append(
                f"- {NOISE_LABELS[noise]}, rho={rho:.1f}: {METHOD_LABELS[best]} MSE={best_mse:.4g} vs Algorithm 1 MSE={alg_mse:.4g}."
            )
    else:
        lines.append("- No deployable method beat Algorithm 1 by at least 10% MSE in this run.")
    lines += ["", "## Caveats", ""]
    if not cargo_available:
        lines.append("- `cargo` was not on PATH, so the Rust tests and the Rust simulation were not rerun with this report.")
    lines += [
        "- The Student-t(3) empirical variance diagnostic is necessarily noisy because the fourth moment is infinite; the script uses a wider tolerance for that diagnostic.",
        "- Bootstrap-UCB uses the final denominator safeguard `max(u_d, 1e-12)` only after the lower bootstrap quantiles are positive; no negative ICC estimates are removed before quantiles.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused robustness experiment for disattenuation Algorithm 1.")
    parser.add_argument("--repetitions", type=int, default=4000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20270720)
    parser.add_argument(
        "--noise-types",
        nargs="+",
        default=["gaussian", "student_t3", "contaminated"],
        choices=["gaussian", "student_t3", "contaminated"],
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions <= 0 or args.bootstrap_repetitions <= 0:
        raise SystemExit("--repetitions and --bootstrap-repetitions must be positive")

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "results" / f"disatt_robustness_{stamp}_reproduced"
    elif not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    diag_rng = np.random.default_rng(args.seed ^ 0xD1A6)

    started = time.perf_counter()
    variance_diagnostics = run_noise_diagnostics(diag_rng, args.noise_types)
    gaussian_consistency = run_gaussian_consistency_diagnostic(diag_rng)

    accum: dict[tuple[str, float, str], Accumulator] = {
        (noise, rho, method): Accumulator()
        for noise in args.noise_types
        for rho in RHO_GRID
        for method in METHOD_ORDER
    }

    for noise in args.noise_types:
        for rho in RHO_GRID:
            for rep in range(args.repetitions):
                x, y, _ = simulate_dataset(rng, rho, noise)
                est = estimate(x, y)
                for method in METHOD_ORDER:
                    t0 = time.perf_counter()
                    if method == "bootstrap_ucb":
                        result = bootstrap_ucb(est, rng, ALPHA, args.bootstrap_repetitions)
                    elif method == "naive_plugin":
                        result = naive_plugin(est)
                    elif method == "clipped_plugin":
                        result = clipped_plugin(est)
                    elif method == "fixed_floor_0.1":
                        result = fixed_floor(est, 0.1)
                    elif method == "clipping_only":
                        result = clipped_plugin(est)
                    elif method == "gate_point":
                        result = gate_point(est, ALPHA)
                    elif method == "ucb_no_gate":
                        result = ucb_no_gate(est, ALPHA)
                    elif method == "algorithm1":
                        result = algorithm1(est, rho, ALPHA)
                    else:
                        raise AssertionError(method)
                    elapsed = time.perf_counter() - t0
                    if method != "naive_plugin":
                        if result.value < -1.0 - 1e-12 or result.value > 1.0 + 1e-12:
                            raise AssertionError(f"unclipped output for {method}: {result.value}")
                    accum[(noise, rho, method)].add(result, elapsed)
                if (rep + 1) % max(1, args.repetitions // 4) == 0:
                    print(f"{noise} rho={rho:.1f}: {rep + 1}/{args.repetitions}")

    rows = [
        accum[(noise, rho, method)].row(noise, rho, method, args.repetitions)
        for noise in args.noise_types
        for rho in RHO_GRID
        for method in METHOD_ORDER
    ]

    expected_cells = len(args.noise_types) * len(RHO_GRID) * len(METHOD_ORDER)
    if len(rows) != expected_cells:
        raise AssertionError(f"missing cells: {len(rows)} != {expected_cells}")

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "noise_variance_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(variance_diagnostics[0].keys()))
        writer.writeheader()
        writer.writerows(variance_diagnostics)

    config = {
        "script": "scripts/disatt_robustness.py",
        "seed": args.seed,
        "repetitions": args.repetitions,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "noise_types": args.noise_types,
        "n": N,
        "k": K,
        "r_true": R_TRUE,
        "alpha": ALPHA,
        "rho_grid": RHO_GRID,
        "sigma2_parameterization": "sigma^2 = K * (1 - rho) / rho for K-replicate mean reliability",
        "bootstrap_denominator_safeguard": f"max(u_d, {DENOM_EPS}) after positive lower bootstrap quantiles",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    diagnostics = {
        "noise_variance": variance_diagnostics,
        "gaussian_consistency": gaussian_consistency,
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    make_main_figure(rows, output_dir / "robustness_mse.pdf")
    make_ablation_table(rows, output_dir / "ablation_mse.md")
    elapsed = time.perf_counter() - started
    write_report(rows, config, diagnostics, output_dir, elapsed)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
