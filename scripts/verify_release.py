"""Check the release: required files, data shapes, stored results and file hashes.

Run from the repository root:
    uv run python scripts/verify_release.py

The checks are:
1. required files are present, and every figure script has its PDF in figures/;
2. the LLM correctness tensors have the expected models, shapes and 0/1 values;
3. the stored Algorithm 2 results reproduce the reported counts and rates;
4. the stored robustness sweep has the expected cells and values;
5. the held-out wording summary matches PROMPTS.md;
6. every file listed in MANIFEST.json is present with the recorded SHA-256.
A figure PDF regenerated with figscripts/ embeds a new creation date, so a hash
difference under figures/ is reported but does not fail the check.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from release_files import MANIFEST, ROOT, digest, release_files  # noqa: E402

RESULTS = ROOT / "results" / "algo2_reported"
EXCLUDED_EXTRA_MODELS = {
    "Falcon3-7B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2-7B-Instruct",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tolerance: float = 5e-4) -> None:
    require(abs(actual - expected) <= tolerance, f"{actual} != {expected}")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_required_files() -> None:
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "REPRODUCIBILITY.md",
        "ENVIRONMENT.md",
        "KNOWN_LIMITATIONS.md",
        "PROMPTS.md",
        "MANIFEST.json",
        "pyproject.toml",
        "uv.lock",
        "Cargo.toml",
        "crates/erel-core/src/disatt.rs",
        "crates/erel-cli/src/main.rs",
        "scripts/disatt_robustness.py",
        "scripts/interval_compare.py",
        "scripts/prop2_check.py",
        "scripts/real_recover.py",
        "scripts/coverage_hetero.py",
        "scripts/cond_coverage.py",
        "scripts/boot_calibrate.py",
        "scripts/icc_bias.py",
        "scripts/binary_gtheory.py",
        "scripts/validate_decisions.py",
        "scripts/shot_format.py",
        "scripts/holdout_wordings.py",
        "scripts/algo2_simrank/run.py",
        "scripts/build_manifest.py",
        "scripts/release_files.py",
        "figscripts/style.py",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    require(not missing, f"missing required files: {missing}")
    scripts = sorted((ROOT / "figscripts").glob("fig*.py"))
    require(len(scripts) == 10, f"expected 10 figure scripts, found {len(scripts)}")
    missing_pdfs = [p.stem for p in scripts if not (ROOT / "figures" / f"{p.stem}.pdf").is_file()]
    require(not missing_pdfs, f"figure scripts without a PDF in figures/: {missing_pdfs}")


def tensor_model_names(prefix: str) -> set[str]:
    names = set()
    for path in (ROOT / "data").glob(f"{prefix}*.npy"):
        names.add(path.stem.removeprefix(prefix))
    return names


def verify_tensors() -> None:
    arc = tensor_model_names("llm_decomp_")
    arc = {name for name in arc if not name.startswith(("hellaswag_", "mmlu_", "openbookqa_"))}
    require(len(arc) == 15, f"ARC model count is {len(arc)}, expected 15")
    for benchmark in ("hellaswag", "mmlu", "openbookqa"):
        names = tensor_model_names(f"llm_decomp_{benchmark}_")
        require(len(names) == 11, f"{benchmark} model count is {len(names)}, expected 11")
        require(not (names & EXCLUDED_EXTRA_MODELS), f"{benchmark} contains ARC-only models")
    for path in (ROOT / "data").glob("llm_decomp_*.npy"):
        tensor = np.load(path, mmap_mode="r")
        require(tensor.shape == (4, 4, 2, 2365), f"unexpected tensor shape: {path.name} {tensor.shape}")
        require(np.isfinite(tensor).all(), f"non-finite tensor: {path.name}")
        require(((tensor == 0) | (tensor == 1)).all(), f"non-binary tensor: {path.name}")


def verify_algorithm2() -> None:
    summary = load_json(RESULTS / "summary.json")
    expected = {
        "arc": (15, 105, 3.5125295856592067, 93, 5, 7),
        "hellaswag": (11, 55, 3.336063658373172, 42, 4, 9),
        "mmlu": (11, 55, 3.3440958303500143, 43, 3, 9),
        "openbookqa": (11, 55, 3.380031130513001, 43, 4, 8),
    }
    certified_total = protocol_total = pair_total = pooled_total = 0
    for benchmark, (models, pairs, q, certified, protocol, refused) in expected.items():
        row = summary["benchmarks"][benchmark]
        counts = row["verdict_counts"]
        got_certified = counts.get("stable-a", 0) + counts.get("stable-b", 0)
        got_refused = counts.get("unresolved", 0) + counts.get("unresolved-mixed", 0)
        require((row["n_models"], row["n_pairs"]) == (models, pairs), benchmark)
        close(row["critical_value"], q, 1e-12)
        require((got_certified, counts.get("protocol-dependent", 0), got_refused) == (certified, protocol, refused), benchmark)
        certified_total += got_certified
        protocol_total += counts.get("protocol-dependent", 0)
        pair_total += pairs
        pooled_total += summary["pooled_baselines"][benchmark]["resolved_pairs"]
    require((certified_total, protocol_total, pair_total) == (221, 16, 270), "Algorithm 2 totals")
    require(pooled_total == 254, f"pooled resolved total {pooled_total}")

    extended = load_json(RESULTS / "extended_validation.json")
    fwer_expected = {"arc": 0.065, "hellaswag": 0.06, "mmlu": 0.07, "openbookqa": 0.05}
    wording_expected = {"arc": (11, 555), "hellaswag": (4, 257), "mmlu": (6, 251), "openbookqa": (13, 261)}
    order_expected = {"arc": (0, 553), "hellaswag": (0, 248), "mmlu": (1, 246), "openbookqa": (0, 258)}
    for benchmark in fwer_expected:
        close(extended["global_permutation_null"][benchmark]["empirical_fwer"], fwer_expected[benchmark], 1e-12)
        wording = extended["heldout_transfer"][benchmark]["wording"]
        order = extended["heldout_transfer"][benchmark]["order"]
        require((wording["stable_failures"], wording["stable"]) == wording_expected[benchmark], benchmark)
        require((order["stable_failures"], order["stable"]) == order_expected[benchmark], benchmark)

    decision = load_json(RESULTS / "decision_validation.json")["scenarios"]
    close(decision["stable"]["protocol-aware-max-t"]["stable_detection_rate"], 0.8952, 1e-12)
    close(decision["near-tie-stable"]["protocol-aware-max-t"]["stable_detection_rate"], 0.04533333333333334, 1e-12)
    close(decision["protocol-reversal"]["protocol-aware-max-t"]["reversal_detection_rate"], 0.6833333333333333, 1e-12)
    close(decision["protocol-reversal"]["pooled-max-t"]["false_stable_rate"], 0.6455333333333333, 1e-12)
    close(decision["global-null"]["protocol-aware-max-t"]["null_family_fwer"], 0.051, 1e-12)
    close(decision["global-null"]["pooled-max-t"]["null_family_fwer"], 0.043, 1e-12)

    cross = load_json(RESULTS / "cross_harness_native.json")
    stable = sum(cross[b]["stable"] for b in expected)
    pooled = sum(cross[b]["pooled_stable"] for b in expected)
    total = sum(cross[b]["comparisons"] for b in expected)
    require((stable, pooled, total) == (342, 1797, 9600), "cross-harness totals")


def verify_robustness() -> None:
    path = ROOT / "results" / "disatt_robustness_full_4000x500_seed20270720" / "results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 120, f"robustness row count {len(rows)}")
    for row in rows:
        require(math.isfinite(float(row["mse"])), "non-finite robustness MSE")
    index = {(r["noise"], float(r["rho"]), r["method"]): r for r in rows}
    close(float(index[("gaussian", 0.4, "algorithm1")]["mse"]), 0.07534038544766562, 1e-12)
    close(float(index[("student_t3", 0.4, "algorithm1")]["mse"]), 0.07107057308997355, 1e-12)
    close(float(index[("contaminated", 0.4, "algorithm1")]["mse"]), 0.07094645043961302, 1e-12)
    gaussian_freqs = [
        float(index[("gaussian", rho, "algorithm1")]["alg1_conservative_frequency"])
        for rho in (0.1, 0.2, 0.4, 0.6, 0.8)
    ]
    require(min(gaussian_freqs) >= 0.8, f"conservative frequency {gaussian_freqs}")


def verify_prompts_and_controls() -> None:
    holdout = load_json(ROOT / "data" / "holdout_wordings.json")
    require((holdout["checked"], holdout["failures"]) == (93, 1), "held-out summary")
    require(holdout["cfg_flips"] == [5, 744], "held-out configuration flips")
    require(holdout["pd_replicated"] == [5, 5], "held-out protocol patterns")
    require(len(holdout["wordings"]) == 4, "four held-out prompts")
    prompt_text = (ROOT / "PROMPTS.md").read_text(encoding="utf-8")
    for wording in holdout["wordings"]:
        require(wording in prompt_text, "held-out wording missing from PROMPTS.md")


def verify_manifest() -> tuple[int, list[str], list[str]]:
    manifest = load_json(MANIFEST)
    entries = manifest["files"]
    require(manifest["file_count"] == len(entries), "MANIFEST.json file_count does not match its file list")
    listed = set()
    missing, changed, figures_changed = [], [], []
    for entry in entries:
        relative = entry["path"]
        listed.add(relative)
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        size, sha = digest(path)
        if (size, sha) != (entry["bytes"], entry["sha256"]):
            (figures_changed if relative.startswith("figures/") else changed).append(relative)
    require(not missing, f"files listed in MANIFEST.json are missing: {missing}")
    require(not changed, f"files differ from MANIFEST.json: {changed}")
    unlisted = [p.relative_to(ROOT).as_posix() for p in release_files() if p.relative_to(ROOT).as_posix() not in listed]
    return len(entries), figures_changed, unlisted


def main() -> None:
    verify_required_files()
    verify_tensors()
    verify_algorithm2()
    verify_robustness()
    verify_prompts_and_controls()
    count, figures_changed, unlisted = verify_manifest()
    for relative in figures_changed:
        print(f"note: {relative} differs from the released bytes (regenerated figure?)")
    for relative in unlisted:
        print(f"note: {relative} is not listed in MANIFEST.json")
    print(f"PASS: file structure, 11-model design, stored results, prompts, and {count} manifest hashes")


if __name__ == "__main__":
    main()
