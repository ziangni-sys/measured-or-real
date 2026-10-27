from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from .core import analyze_scores


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate simultaneous paired ranking outputs")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "algo2_reproduced")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--items", type=int, default=600)
    parser.add_argument("--models", type=int, default=6)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20270723)
    return parser.parse_args()


def null_fwer(args: argparse.Namespace) -> tuple[int, list[float]]:
    """Estimate FWER when all models share the same item-dependent accuracy."""
    rng = np.random.default_rng(args.seed)
    false_families = 0
    critical_values = []
    names = [f"null-model-{j}" for j in range(args.models)]
    for simulation in range(args.simulations):
        item_effect = rng.normal(0.0, 1.0, size=(2, args.items))
        probabilities = 1.0 / (1.0 + np.exp(-item_effect))
        scores = rng.random((args.models, 2, args.items)) < probabilities[None, :, :]
        result = analyze_scores(
            scores,
            names,
            alpha=args.alpha,
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed + simulation + 1,
        )
        critical_values.append(float(result["critical_value"]))
        any_false = any(
            row["lower_protocol_0"] > 0.0
            or row["upper_protocol_0"] < 0.0
            or row["lower_protocol_1"] > 0.0
            or row["upper_protocol_1"] < 0.0
            for row in result["pairs"]
        )
        false_families += int(any_false)
    return false_families, critical_values


def rank_widths(results_dir: Path) -> dict[str, dict[str, dict[str, float | int]]]:
    with (results_dir / "ranks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        output[benchmark] = {}
        for target in sorted({row["target"] for row in rows if row["benchmark"] == benchmark}):
            selected = [row for row in rows if row["benchmark"] == benchmark and row["target"] == target]
            widths = np.array([int(row["rank_upper"]) - int(row["rank_lower"]) for row in selected])
            output[benchmark][target] = {
                "mean_width": float(widths.mean()),
                "max_width": int(widths.max()),
                "singleton_count": int(np.sum(widths == 0)),
            }
    return output


def render_report(payload: dict) -> str:
    config = payload["null_simulation"]
    lines = [
        "# Validation of Simultaneous Paired Ranking",
        "",
        "## Global-null simulation",
        "",
        f"All {config['models']} synthetic models share the same item-dependent success probability in both protocols. Over {config['simulations']} independent families, at least one false pair/protocol resolution occurred in {config['false_families']} families ({config['empirical_fwer']:.3f}) at nominal alpha={config['alpha']:.3f}. The Monte Carlo standard error is {config['fwer_mc_se']:.3f}.",
        "",
        "## Rank-set summary",
        "",
        "| Benchmark | Target | Mean width | Maximum width | Singleton sets |",
        "|---|---|---:|---:|---:|",
    ]
    for benchmark, targets in payload["rank_widths"].items():
        for target, row in targets.items():
            lines.append(
                f"| {benchmark} | {target} | {row['mean_width']:.2f} | "
                f"{row['max_width']} | {row['singleton_count']} |"
            )
    lines.extend(
        [
            "",
            "## Scope and companion validation",
            "",
            "This simulation validates global-null family behavior only. Known-rank coverage, real-data permutation, held-out wording/order transfer, and cross-harness self-pair checks are reported in `EXTENDED_VALIDATION.md`; known-truth typed-decision comparisons are in `DECISION_VALIDATION.md`. None of these establishes coverage for an unrestricted future prompt population.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    if not (results_dir / "summary.json").exists():
        raise SystemExit(f"missing completed results in {results_dir}")
    false_families, critical_values = null_fwer(args)
    empirical_fwer = false_families / args.simulations
    payload = {
        "null_simulation": {
            "simulations": args.simulations,
            "items": args.items,
            "models": args.models,
            "protocols": 2,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "alpha": args.alpha,
            "seed": args.seed,
            "false_families": false_families,
            "empirical_fwer": empirical_fwer,
            "fwer_mc_se": float(np.sqrt(empirical_fwer * (1.0 - empirical_fwer) / args.simulations)),
            "mean_critical_value": float(np.mean(critical_values)),
        },
        "rank_widths": rank_widths(results_dir),
    }
    (results_dir / "validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (results_dir / "VALIDATION.md").write_text(render_report(payload), encoding="utf-8")
    print(
        f"global-null FWER: {false_families}/{args.simulations} = {empirical_fwer:.3f}; "
        f"wrote {results_dir / 'VALIDATION.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
