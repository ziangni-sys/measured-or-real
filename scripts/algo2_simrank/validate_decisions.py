from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .core import _resolved_winner, analyze_pooled_scores, analyze_scores, pair_indices


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("pooled-max-t", "protocol-aware-max-t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate typed fixed-grid model-comparison decisions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "algo2_reproduced")
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--items", type=int, default=600)
    parser.add_argument("--models", type=int, default=6)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20270730)
    return parser.parse_args()


def scenario_abilities(n_models: int) -> dict[str, np.ndarray]:
    if n_models < 3:
        raise ValueError("at least three models are required")
    base = np.linspace(0.75, -0.75, n_models)
    near = np.linspace(0.12, -0.12, n_models)
    return {
        "global-null": np.zeros((n_models, 2), dtype=np.float64),
        "stable": np.stack([base, 0.8 * base], axis=1),
        "protocol-reversal": np.stack([0.45 * base, -base], axis=1),
        "near-tie-stable": np.stack([near, 0.8 * near], axis=1),
    }


def truth_verdicts(abilities: np.ndarray) -> list[str]:
    output = []
    for a, b in pair_indices(abilities.shape[0]):
        differences = abilities[a] - abilities[b]
        if np.all(differences > 0.0):
            output.append("stable-a")
        elif np.all(differences < 0.0):
            output.append("stable-b")
        elif differences[0] * differences[1] < 0.0:
            output.append("protocol-dependent")
        else:
            output.append("unresolved")
    return output


def method_decisions(
    scores: np.ndarray,
    names: list[str],
    *,
    alpha: float,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, tuple[list[str], list[bool]]]:
    n_models, n_protocols, n_items = scores.shape
    if n_protocols != 2:
        raise ValueError("exactly two protocols are required")
    pooled = analyze_pooled_scores(
        scores,
        names,
        alpha=alpha,
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed + 1_000_000,
    )
    pooled_verdicts = [str(row["verdict"]) for row in pooled["pairs"]]
    pooled_resolved = [int(row["winner"]) != 0 for row in pooled["pairs"]]

    max_t = analyze_scores(
        scores,
        names,
        alpha=alpha,
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
    max_t_verdicts = [str(row["verdict"]) for row in max_t["pairs"]]
    max_t_resolved = [
        any(
            _resolved_winner(float(row[f"lower_protocol_{protocol}"]), float(row[f"upper_protocol_{protocol}"]))
            != 0
            for protocol in range(2)
        )
        for row in max_t["pairs"]
    ]

    return {
        "pooled-max-t": (pooled_verdicts, pooled_resolved),
        "protocol-aware-max-t": (max_t_verdicts, max_t_resolved),
    }


def _empty_counts() -> dict[str, int]:
    return {
        "pairs": 0,
        "exact": 0,
        "refused": 0,
        "false_stable": 0,
        "stable_total": 0,
        "stable_detected": 0,
        "reversal_total": 0,
        "reversal_detected": 0,
        "null_total": 0,
        "null_resolved": 0,
        "families_with_any_null_resolution": 0,
    }


def run_validation(args: argparse.Namespace) -> dict:
    if args.simulations <= 0 or args.items < 2 or args.bootstrap_repetitions <= 0:
        raise ValueError("simulation, item, and bootstrap counts must be positive")
    abilities_by_scenario = scenario_abilities(args.models)
    names = [f"sim-model-{index}" for index in range(args.models)]
    rng = np.random.default_rng(args.seed)
    output: dict[str, dict[str, dict]] = {}

    for scenario_index, (scenario, abilities) in enumerate(abilities_by_scenario.items()):
        truth = truth_verdicts(abilities)
        counts = {method: _empty_counts() for method in METHODS}
        for repetition in range(args.simulations):
            shared_difficulty = rng.normal(0.0, 1.0, size=args.items)
            protocol_noise = rng.normal(0.0, 0.3, size=(2, args.items))
            logits = abilities[:, :, None] + shared_difficulty[None, None, :] + protocol_noise[None, :, :]
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            scores = (rng.random(probabilities.shape) < probabilities).astype(np.float64)
            decisions = method_decisions(
                scores,
                names,
                alpha=args.alpha,
                bootstrap_repetitions=args.bootstrap_repetitions,
                seed=args.seed + 1 + scenario_index * args.simulations + repetition,
            )
            for method, (predictions, any_resolved) in decisions.items():
                row = counts[method]
                family_null_resolution = False
                for expected, predicted, resolved in zip(truth, predictions, any_resolved):
                    row["pairs"] += 1
                    row["exact"] += int(predicted == expected)
                    row["refused"] += int(predicted.startswith("unresolved"))
                    row["false_stable"] += int(predicted.startswith("stable-") and predicted != expected)
                    if expected.startswith("stable-"):
                        row["stable_total"] += 1
                        row["stable_detected"] += int(predicted == expected)
                    elif expected == "protocol-dependent":
                        row["reversal_total"] += 1
                        row["reversal_detected"] += int(predicted == expected)
                    else:
                        row["null_total"] += 1
                        row["null_resolved"] += int(resolved)
                        family_null_resolution = family_null_resolution or resolved
                row["families_with_any_null_resolution"] += int(family_null_resolution)

        output[scenario] = {}
        for method, row in counts.items():
            pairs = row["pairs"]
            output[scenario][method] = {
                **row,
                "exact_verdict_rate": row["exact"] / pairs,
                "refusal_rate": row["refused"] / pairs,
                "false_stable_rate": row["false_stable"] / pairs,
                "stable_detection_rate": (
                    row["stable_detected"] / row["stable_total"] if row["stable_total"] else None
                ),
                "reversal_detection_rate": (
                    row["reversal_detected"] / row["reversal_total"] if row["reversal_total"] else None
                ),
                "null_pair_resolution_rate": (
                    row["null_resolved"] / row["null_total"] if row["null_total"] else None
                ),
                "null_family_fwer": (
                    row["families_with_any_null_resolution"] / args.simulations if row["null_total"] else None
                ),
            }

    return {
        "config": {
            "simulations": args.simulations,
            "items": args.items,
            "models": args.models,
            "protocols": 2,
            "alpha": args.alpha,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
            "target": "typed model-comparison verdicts conditional on a fixed configuration grid",
        },
        "scenarios": output,
    }


def render_report(payload: dict) -> str:
    config = payload["config"]
    lines = [
        "# Fixed-Grid Decision Validation",
        "",
        "The simulation compares typed model-comparison decisions when the item-population truth is known. "
        "It conditions on a fixed configuration grid and does not assess transfer to unseen prompt populations.",
        "",
        f"Settings: {config['simulations']} families per scenario, {config['models']} models, "
        f"{config['items']} paired items, alpha={config['alpha']:.3f}, and "
        f"{config['bootstrap_repetitions']} Gaussian max-t draws.",
        "",
        "| Scenario | Method | Exact typed verdict | Stable detection | Reversal detection | False stable | Refusal | Null-family FWER |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, methods in payload["scenarios"].items():
        for method in METHODS:
            row = methods[method]

            def percent(value: float | None) -> str:
                return "--" if value is None else f"{100 * value:.1f}%"

            lines.append(
                f"| {scenario} | {method} | {percent(row['exact_verdict_rate'])} | "
                f"{percent(row['stable_detection_rate'])} | {percent(row['reversal_detection_rate'])} | "
                f"{percent(row['false_stable_rate'])} | {percent(row['refusal_rate'])} | "
                f"{percent(row['null_family_fwer'])} |"
            )
    lines.extend(
        [
            "",
            "`Exact typed verdict` requires the correct direction and class (`stable`, `protocol-dependent`, or "
            "`unresolved`). A refusal is not counted as a false stable winner. Pooled max-t controls simultaneous "
            "error for its averaged estimand but cannot emit a "
            "protocol-dependent verdict, so its behavior in the reversal scenario quantifies the information lost by pooling.",
            "",
            "These simulations establish behavior for the stated synthetic item model only. They complement, rather than "
            "replace, the real-data permutation, held-out-configuration, and cross-harness diagnostics.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    payload = run_validation(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "decision_validation.json"
    report_path = output_dir / "DECISION_VALIDATION.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
