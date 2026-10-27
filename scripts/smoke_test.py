"""Run shortened versions of the CPU entry points and write a short report.

Run from the repository root:
    uv run python scripts/smoke_test.py

Outputs go to results/_smoke/ (git-ignored). The intermediate outputs are
deleted at the end; the report stays at results/_smoke/SMOKE_TEST_REPORT.md
unless --report names another file. Figure scripts write their PDFs into the
smoke folder, so figures/ is not touched.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "results" / "_smoke"
MISSING_INPUT = {
    "real_disatt.py": "The CIFAR-100 label vector is absent; see KNOWN_LIMITATIONS.md.",
    "gauge_probe2.py": "The three embedding feature matrices are absent; see KNOWN_LIMITATIONS.md.",
}


def sanitize(text: str) -> str:
    python_env = Path(sys.executable).resolve().parents[1]
    return (
        text.replace(str(ROOT), ".")
        .replace(str(python_env), "<PYTHON_ENV>")
        .replace(str(Path.home()), "<HOME>")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=SMOKE / "SMOKE_TEST_REPORT.md")
    args = parser.parse_args()
    work = SMOKE / "run"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    rel = "results/_smoke/run"

    py = sys.executable
    tests = [
        ("disatt_robustness.py", [py, "scripts/disatt_robustness.py", "--repetitions", "3", "--bootstrap-repetitions", "5", "--seed", "7", "--output-dir", f"{rel}/robustness"], False),
        ("interval_compare.py", [py, "scripts/interval_compare.py", "--repetitions", "3", "--seed", "7"], False),
        ("prop2_check.py", [py, "scripts/prop2_check.py", "--repetitions", "5", "--large-repetitions", "100", "--seed", "7"], False),
        ("real_recover.py", [py, "scripts/real_recover.py"], False),
        ("coverage_hetero.py", [py, "scripts/coverage_hetero.py", "--repetitions", "3", "--seed", "7"], False),
        ("cond_coverage.py", [py, "scripts/cond_coverage.py", "--repetitions", "3", "--seed", "7"], False),
        ("boot_calibrate.py", [py, "scripts/boot_calibrate.py", "--repetitions", "2", "--bootstrap-repetitions", "3", "--seed", "7"], False),
        ("icc_bias.py", [py, "scripts/icc_bias.py", "--repetitions", "3", "--seed", "7"], False),
        ("binary_gtheory.py", [py, "scripts/binary_gtheory.py"], False),
        ("validate_decisions.py", [py, "scripts/validate_decisions.py", "--output-dir", f"{rel}/decisions", "--simulations", "2", "--items", "100", "--bootstrap-repetitions", "100", "--seed", "7"], False),
        ("shot_format.py", [py, "scripts/shot_format.py", "--analyze-only"], False),
        ("holdout_wordings.py", [py, "scripts/holdout_wordings.py", "--analyze-only", "--pairwise-path", "results/algo2_reported/pairwise.csv", "--output", f"{rel}/holdout_wordings.json"], False),
        ("Algorithm 2 alpha=0.01", [py, "-m", "scripts.algo2_simrank.run", "--alpha", "0.01", "--bootstrap-repetitions", "100", "--seed", "7", "--output-dir", f"{rel}/a001"], False),
        ("Algorithm 2 alpha=0.05", [py, "-m", "scripts.algo2_simrank.run", "--alpha", "0.05", "--bootstrap-repetitions", "100", "--seed", "7", "--output-dir", f"{rel}/a005"], False),
        ("Algorithm 2 alpha=0.1", [py, "-m", "scripts.algo2_simrank.run", "--alpha", "0.1", "--bootstrap-repetitions", "100", "--seed", "7", "--output-dir", f"{rel}/a01"], False),
        ("Algorithm 2 unit tests", [py, "-m", "unittest", "discover", "-s", "scripts/algo2_simrank/tests", "-v"], False),
    ]
    for script in sorted((ROOT / "figscripts").glob("fig*.py")):
        tests.append((f"figscripts/{script.name}", [py, f"figscripts/{script.name}"], False))
    # These two scripts may fail only while their inputs are absent (KNOWN_LIMITATIONS.md).
    data = ROOT / "data"
    labels_absent = not (data / "cifar100_labels.npy").exists() and not any(
        (data / name / "train.npz").exists() for name in ("cifar100_hf", "cifar_hf")
    )
    features_absent = not all((data / "gauge_feats" / f"feat_seed{s}_e20.npy").exists() for s in range(3))
    tests += [
        ("real_disatt.py", [py, "scripts/real_disatt.py"], labels_absent),
        ("gauge_probe2.py", [py, "scripts/gauge_probe2.py"], features_absent),
    ]
    cargo = shutil.which("cargo")
    if cargo:
        tests.append(("erel ablate", [cargo, "run", "--release", "-p", "erel-cli", "--", "ablate"], False))
    else:
        tests.append(("erel ablate", ["cargo", "run", "--release", "-p", "erel-cli", "--", "ablate"], True))

    env = dict(os.environ)
    env["FIG_OUT"] = str(work / "figures")
    env["FIG_PREVIEW"] = str(work / "figures" / "preview")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    rows = []
    unexpected = []
    for name, command, expected_failure in tests:
        started = time.perf_counter()
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
            code = result.returncode
            output = sanitize((result.stdout + "\n" + result.stderr).strip())
        except (OSError, subprocess.TimeoutExpired) as error:
            code = -1
            output = sanitize(str(error))
        elapsed = time.perf_counter() - started
        passed = code == 0
        if expected_failure and not passed:
            if name in MISSING_INPUT:
                # Only a stop on the missing input is the documented limitation; any other error is a failure.
                if "FileNotFoundError" in output or "missing" in output:
                    output = MISSING_INPUT[name]
                else:
                    expected_failure = False
            elif name == "erel ablate":
                output = "cargo is not installed on this machine."
        status = "PASS" if passed else ("EXPECTED LIMITATION" if expected_failure else "FAIL")
        rows.append((name, status, elapsed, output[-500:]))
        print(f"{status:20s} {elapsed:7.2f}s  {name}", flush=True)
        if not passed and not expected_failure:
            unexpected.append(name)

    lines = [
        "# Smoke Test Report",
        "",
        "| Entry point | Status | Seconds |",
        "|---|---|---:|",
    ]
    for name, status, elapsed, _ in rows:
        lines.append(f"| `{name}` | {status} | {elapsed:.2f} |")
    lines.extend(["", "## Failure Tails", ""])
    for name, status, _, output in rows:
        if status != "PASS":
            lines.extend([f"### {name}: {status}", "", "```text", output, "```", ""])
    lines.extend(
        [
            "The two data-dependent expected limitations are documented in `KNOWN_LIMITATIONS.md`.",
            "The Rust check needs a Rust toolchain on PATH.",
            "",
        ]
    )
    report = args.report if args.report.is_absolute() else ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    shutil.rmtree(work)
    if unexpected:
        raise SystemExit(f"unexpected smoke-test failures: {unexpected}")
    try:
        shown = report.relative_to(ROOT).as_posix()
    except ValueError:
        shown = report.name
    print(f"wrote {shown}")


if __name__ == "__main__":
    main()
