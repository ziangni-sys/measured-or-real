from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .core import _pair_verdict, analyze_scores
from .data import load_benchmark


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("arc", "hellaswag", "mmlu", "openbookqa")
PROTOCOL_TARGETS = ("zero-shot", "few-shot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended validation for simultaneous paired ranking")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "algo2_reproduced")
    parser.add_argument("--rank-simulations", type=int, default=200)
    parser.add_argument("--global-null-repetitions", type=int, default=200)
    parser.add_argument("--pair-null-repetitions", type=int, default=200)
    parser.add_argument("--self-repetitions", type=int, default=200)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--transfer-bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--simulation-items", type=int, default=600)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20270724)
    return parser.parse_args()


def point_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.array([1 + np.sum(values > value) for value in values], dtype=np.int64)


def interval_winner(lower: float, upper: float) -> int:
    if lower > 0.0:
        return 1
    if upper < 0.0:
        return -1
    return 0


def known_rank_coverage(args: argparse.Namespace) -> dict:
    """Check simultaneous rank coverage when protocol-specific true ranks are known."""
    n_models = 6
    base = np.linspace(1.0, -1.0, n_models)
    abilities = np.stack([base, np.roll(base, 1)], axis=1)
    true_ranks = np.stack([point_ranks(abilities[:, s]) for s in range(2)], axis=0)
    names = [f"sim-model-{index}" for index in range(n_models)]
    rng = np.random.default_rng(args.seed)
    family_covered = 0
    model_protocol_covered = 0
    total_rank_width = 0
    total_rank_sets = args.rank_simulations * n_models * 2

    for repetition in range(args.rank_simulations):
        item_effects = rng.normal(0.0, 1.0, size=(2, args.simulation_items))
        logits = abilities[:, :, None] + item_effects[None, :, :]
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        scores = rng.random(probabilities.shape) < probabilities
        result = analyze_scores(
            scores,
            names,
            alpha=args.alpha,
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed + repetition + 1,
        )
        rows = {(row["target"], int(row["model_index"])): row for row in result["ranks"]}
        covered_this_family = True
        for protocol, target in enumerate(PROTOCOL_TARGETS):
            for model in range(n_models):
                row = rows[(target, model)]
                rank = int(true_ranks[protocol, model])
                covered = int(row["rank_lower"]) <= rank <= int(row["rank_upper"])
                model_protocol_covered += int(covered)
                covered_this_family = covered_this_family and covered
                total_rank_width += int(row["rank_upper"]) - int(row["rank_lower"])
        family_covered += int(covered_this_family)

    family_rate = family_covered / args.rank_simulations
    return {
        "simulations": args.rank_simulations,
        "models": n_models,
        "items": args.simulation_items,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "true_abilities": abilities.tolist(),
        "family_covered": family_covered,
        "simultaneous_rank_coverage": family_rate,
        "coverage_mc_se": float(np.sqrt(family_rate * (1.0 - family_rate) / args.rank_simulations)),
        "marginal_model_protocol_coverage": model_protocol_covered / total_rank_sets,
        "mean_rank_set_width": total_rank_width / total_rank_sets,
    }


def global_permutation_null(args: argparse.Namespace) -> dict[str, dict]:
    """Permute complete model protocol profiles within item to create a global null."""
    output = {}
    for benchmark_index, benchmark in enumerate(BENCHMARKS):
        names, tensors, _ = load_benchmark(ROOT / "data", benchmark)
        scores = tensors.mean(axis=(1, 2))
        n_models, _, n_items = scores.shape
        by_item = scores.transpose(2, 0, 1)
        rng = np.random.default_rng(args.seed + 10_000 + benchmark_index)
        false_families = 0
        total_resolved_pairs = 0
        total_stable_pairs = 0
        total_protocol_dependent = 0
        for repetition in range(args.global_null_repetitions):
            permutation = np.argsort(rng.random((n_items, n_models)), axis=1)
            shuffled = np.take_along_axis(by_item, permutation[:, :, None], axis=1).transpose(1, 2, 0)
            result = analyze_scores(
                shuffled,
                names,
                alpha=args.alpha,
                bootstrap_repetitions=args.bootstrap_repetitions,
                seed=args.seed + 20_000 + benchmark_index * args.global_null_repetitions + repetition,
            )
            any_resolved = False
            for row in result["pairs"]:
                winners = (
                    interval_winner(row["lower_protocol_0"], row["upper_protocol_0"]),
                    interval_winner(row["lower_protocol_1"], row["upper_protocol_1"]),
                )
                resolved = any(winner != 0 for winner in winners)
                any_resolved = any_resolved or resolved
                total_resolved_pairs += int(resolved)
                total_stable_pairs += int(row["verdict"].startswith("stable-"))
                total_protocol_dependent += int(row["verdict"] == "protocol-dependent")
            false_families += int(any_resolved)
        rate = false_families / args.global_null_repetitions
        output[benchmark] = {
            "repetitions": args.global_null_repetitions,
            "false_families": false_families,
            "empirical_fwer": rate,
            "fwer_mc_se": float(np.sqrt(rate * (1.0 - rate) / args.global_null_repetitions)),
            "resolved_pairs": total_resolved_pairs,
            "stable_pairs": total_stable_pairs,
            "protocol_dependent_pairs": total_protocol_dependent,
            "pair_replicates": args.global_null_repetitions * result["n_pairs"],
        }
    return output


def pairwise_swap_null(args: argparse.Namespace, full_summary: dict) -> dict[str, dict]:
    """Per-pair item-wise swap diagnostic using the deployed max-t q."""
    output = {}
    for benchmark_index, benchmark in enumerate(BENCHMARKS):
        names, tensors, _ = load_benchmark(ROOT / "data", benchmark)
        scores = tensors.mean(axis=(1, 2))
        q = float(full_summary[benchmark]["critical_value"])
        rng = np.random.default_rng(args.seed + 30_000 + benchmark_index)
        counts = {"stable": 0, "protocol-dependent": 0, "unresolved-mixed": 0, "unresolved": 0}
        n_pairs = 0
        for a, b in itertools.combinations(range(len(names)), 2):
            difference = scores[a] - scores[b]
            for _ in range(args.pair_null_repetitions):
                multiplier = np.where(rng.random(scores.shape[-1]) < 0.5, -1.0, 1.0)
                permuted = difference * multiplier[None, :]
                gap = permuted.mean(axis=1)
                se = permuted.std(axis=1, ddof=1) / np.sqrt(permuted.shape[1])
                verdict = _pair_verdict(gap - q * se, gap + q * se)
                key = "stable" if verdict.startswith("stable-") else verdict
                counts[key] += 1
            n_pairs += 1
        output[benchmark] = {
            "repetitions_per_pair": args.pair_null_repetitions,
            "pairs": n_pairs,
            "pair_replicates": n_pairs * args.pair_null_repetitions,
            "critical_value": q,
            **counts,
        }
    return output


def rank_transfer_counts(result: dict, heldout_scores: np.ndarray) -> dict[str, int]:
    rows = {(row["target"], int(row["model_index"])): row for row in result["ranks"]}
    heldout_means = heldout_scores.mean(axis=2)
    protocol_ranks = np.stack([point_ranks(heldout_means[:, s]) for s in range(2)], axis=0)
    reference_weight = float(result["reference_weight"])
    pooled_ranks = point_ranks(
        reference_weight * heldout_means[:, 0] + (1.0 - reference_weight) * heldout_means[:, 1]
    )
    n_models = heldout_scores.shape[0]
    counts = {
        "protocol_covered": 0,
        "protocol_total": 2 * n_models,
        "protocol_family_covered": 1,
        "pooled_covered": 0,
        "pooled_total": n_models,
        "pooled_family_covered": 1,
        "robust_endpoints_covered": 0,
        "robust_endpoints_total": 2 * n_models,
        "robust_family_covered": 1,
    }
    for protocol, target in enumerate(PROTOCOL_TARGETS):
        for model in range(n_models):
            row = rows[(target, model)]
            covered = int(row["rank_lower"]) <= protocol_ranks[protocol, model] <= int(row["rank_upper"])
            counts["protocol_covered"] += int(covered)
            counts["protocol_family_covered"] &= int(covered)
    pooled_target = f"pooled-w={reference_weight:g}"
    robust_target = "weight-robust-[0,1]"
    for model in range(n_models):
        pooled_row = rows[(pooled_target, model)]
        covered = int(pooled_row["rank_lower"]) <= pooled_ranks[model] <= int(pooled_row["rank_upper"])
        counts["pooled_covered"] += int(covered)
        counts["pooled_family_covered"] &= int(covered)
        robust_row = rows[(robust_target, model)]
        for protocol in range(2):
            covered = (
                int(robust_row["rank_lower"])
                <= protocol_ranks[protocol, model]
                <= int(robust_row["rank_upper"])
            )
            counts["robust_endpoints_covered"] += int(covered)
            counts["robust_family_covered"] &= int(covered)
    return counts


def pair_transfer_counts(result: dict, heldout_scores: np.ndarray) -> dict[str, int]:
    heldout_means = heldout_scores.mean(axis=2)
    counts = {"stable": 0, "stable_failures": 0, "protocol_dependent": 0, "pd_replicated": 0}
    for row in result["pairs"]:
        a = int(row["model_a_index"])
        b = int(row["model_b_index"])
        heldout_gap = heldout_means[a] - heldout_means[b]
        if row["verdict"].startswith("stable-"):
            counts["stable"] += 1
            winner = 1 if row["verdict"] == "stable-a" else -1
            counts["stable_failures"] += int(any(np.sign(gap) != winner for gap in heldout_gap))
        elif row["verdict"] == "protocol-dependent":
            counts["protocol_dependent"] += 1
            training_winners = (
                interval_winner(row["lower_protocol_0"], row["upper_protocol_0"]),
                interval_winner(row["lower_protocol_1"], row["upper_protocol_1"]),
            )
            counts["pd_replicated"] += int(
                all(np.sign(heldout_gap[s]) == training_winners[s] for s in range(2))
            )
    return counts


def heldout_transfer(args: argparse.Namespace) -> dict[str, dict]:
    output = {}
    for benchmark_index, benchmark in enumerate(BENCHMARKS):
        names, tensors, _ = load_benchmark(ROOT / "data", benchmark)
        benchmark_output = {}
        for axis_name, axis in (("wording", 1), ("order", 2)):
            totals = {
                "splits": 0,
                "stable": 0,
                "stable_failures": 0,
                "protocol_dependent": 0,
                "pd_replicated": 0,
                "protocol_covered": 0,
                "protocol_total": 0,
                "protocol_family_covered": 0,
                "pooled_covered": 0,
                "pooled_total": 0,
                "pooled_family_covered": 0,
                "robust_endpoints_covered": 0,
                "robust_endpoints_total": 0,
                "robust_family_covered": 0,
            }
            for split_index, selected_tuple in enumerate(itertools.combinations(range(4), 2)):
                selected = np.zeros(4, dtype=bool)
                selected[list(selected_tuple)] = True
                if axis == 1:
                    training_scores = tensors[:, selected].mean(axis=(1, 2))
                    heldout_scores = tensors[:, ~selected].mean(axis=(1, 2))
                else:
                    training_scores = tensors[:, :, selected].mean(axis=(1, 2))
                    heldout_scores = tensors[:, :, ~selected].mean(axis=(1, 2))
                result = analyze_scores(
                    training_scores,
                    names,
                    alpha=args.alpha,
                    bootstrap_repetitions=args.transfer_bootstrap_repetitions,
                    seed=args.seed + 40_000 + benchmark_index * 100 + axis * 10 + split_index,
                )
                for key, value in pair_transfer_counts(result, heldout_scores).items():
                    totals[key] += value
                for key, value in rank_transfer_counts(result, heldout_scores).items():
                    totals[key] += value
                totals["splits"] += 1
            benchmark_output[axis_name] = totals
        output[benchmark] = benchmark_output
    return output


def cross_harness_self_pair(
    args: argparse.Namespace, full_summary: dict, pooled_summary: dict
) -> dict[str, dict]:
    output = {}
    for benchmark_index, benchmark in enumerate(BENCHMARKS):
        names, tensors, _ = load_benchmark(ROOT / "data", benchmark)
        q = float(full_summary[benchmark]["critical_value"])
        pooled_q = float(pooled_summary[benchmark]["critical_value"])
        pooled_weight = float(pooled_summary[benchmark]["protocol_weight"])
        cells = tensors.reshape(len(names), 16, 2, tensors.shape[-1])
        rng = np.random.default_rng(args.seed + 50_000 + benchmark_index)
        counts = {"stable": 0, "protocol-dependent": 0, "unresolved-mixed": 0, "unresolved": 0}
        per_model_stable = {name: 0 for name in names}
        pooled_stable = 0
        pooled_per_model_stable = {name: 0 for name in names}
        for model, name in enumerate(names):
            for _ in range(args.self_repetitions):
                selected = rng.permutation(16)[:8]
                mask = np.zeros(16, dtype=bool)
                mask[selected] = True
                difference = cells[model, mask].mean(axis=0) - cells[model, ~mask].mean(axis=0)
                gap = difference.mean(axis=1)
                se = difference.std(axis=1, ddof=1) / np.sqrt(difference.shape[1])
                verdict = _pair_verdict(gap - q * se, gap + q * se)
                key = "stable" if verdict.startswith("stable-") else verdict
                counts[key] += 1
                per_model_stable[name] += int(key == "stable")
                pooled_difference = (
                    pooled_weight * difference[0] + (1.0 - pooled_weight) * difference[1]
                )
                pooled_gap = float(pooled_difference.mean())
                pooled_se = float(pooled_difference.std(ddof=1) / np.sqrt(difference.shape[1]))
                pooled_winner = interval_winner(
                    pooled_gap - pooled_q * pooled_se, pooled_gap + pooled_q * pooled_se
                )
                pooled_stable += int(pooled_winner != 0)
                pooled_per_model_stable[name] += int(pooled_winner != 0)
        total = len(names) * args.self_repetitions
        output[benchmark] = {
            "models": len(names),
            "repetitions_per_model": args.self_repetitions,
            "comparisons": total,
            "critical_value": q,
            "pooled_critical_value": pooled_q,
            "any_resolution": total - counts["unresolved"],
            "pooled_stable": pooled_stable,
            **counts,
            "per_model_stable": per_model_stable,
            "pooled_per_model_stable": pooled_per_model_stable,
        }
    return output


def render_report(payload: dict) -> str:
    sim = payload["known_rank_simulation"]
    lines = [
        "# Extended Validation of Simultaneous Paired Ranking",
        "",
        "## Known-rank simulation",
        "",
        f"Across {sim['simulations']} independently generated benchmark families, all protocol-specific true ranks were simultaneously contained in their reported rank sets in {sim['family_covered']} runs ({sim['simultaneous_rank_coverage']:.3f}, MC SE {sim['coverage_mc_se']:.3f}) at nominal alpha={payload['config']['alpha']:.3f}. Marginal model-by-protocol rank coverage was {sim['marginal_model_protocol_coverage']:.3f}; mean rank-set width was {sim['mean_rank_set_width']:.2f} positions.",
        "",
        "## Global model-label permutation null",
        "",
        "Each item keeps its complete two-protocol model profile, but profiles are randomly reassigned to model labels. Any resolved pair/protocol ordering is false under this global null.",
        "",
        "| Benchmark | False families | Empirical FWER | MC SE | Resolved pair-replicates |",
        "|---|---:|---:|---:|---:|",
    ]
    for benchmark, row in payload["global_permutation_null"].items():
        lines.append(
            f"| {benchmark} | {row['false_families']}/{row['repetitions']} | {row['empirical_fwer']:.3f} | "
            f"{row['fwer_mc_se']:.3f} | {row['resolved_pairs']}/{row['pair_replicates']} |"
        )
    lines.extend(
        [
            "",
            "## Held-out configuration transfer",
            "",
            "Rank sets are fit on two of four wording/order levels and checked against ranks computed from the held-out two levels. These are transfer diagnostics, not covered by the item-sampling alpha guarantee.",
            "",
            "| Benchmark/axis | Stable direction failures | Protocol rank coverage | Protocol family coverage | Robust endpoint coverage |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for benchmark, axes in payload["heldout_transfer"].items():
        for axis, row in axes.items():
            lines.append(
                f"| {benchmark}/{axis} | {row['stable_failures']}/{row['stable']} | "
                f"{row['protocol_covered']}/{row['protocol_total']} | {row['protocol_family_covered']}/{row['splits']} | "
                f"{row['robust_endpoints_covered']}/{row['robust_endpoints_total']} |"
            )
    lines.extend(
        [
            "",
            "## Cross-harness self-pairs",
            "",
            "Each model is compared with itself using disjoint random halves of the 16 configuration cells. A stable winner is necessarily a harness artifact.",
            "",
            "| Benchmark | Protocol-aware stable | Pooled stable | Any protocol-aware resolution | Total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for benchmark, row in payload["cross_harness_self_pair"].items():
        lines.append(
            f"| {benchmark} | {row['stable']} | {row['pooled_stable']} | "
            f"{row['any_resolution']} | {row['comparisons']} |"
        )
    pair_totals = payload["pairwise_swap_null"]
    lines.extend(
        [
            "",
            "## Pairwise swap diagnostic",
            "",
            "This item-wise pair swap uses the deployed max-t critical value. It is a per-pair diagnostic, not a coherent global-null FWER experiment.",
            "",
            "| Benchmark | Stable false winners | Protocol-dependent | Pair-replicates |",
            "|---|---:|---:|---:|",
        ]
    )
    for benchmark, row in pair_totals.items():
        lines.append(
            f"| {benchmark} | {row['stable']} | {row['protocol-dependent']} | {row['pair_replicates']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The known-rank and permutation experiments assess item-sampling calibration for the fixed-grid target. Held-out and self-pair checks measure sensitivity to the small observed configuration design. Their failures block any claim of coverage for an unrestricted future prompt population; they do not invalidate inference conditional on the prespecified shared grid.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}")
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    full_summary = saved_summary["benchmarks"]
    pooled_summary = saved_summary.get("pooled_baselines")
    missing = [benchmark for benchmark in BENCHMARKS if benchmark not in full_summary]
    if missing:
        raise SystemExit(f"full results are missing benchmarks: {missing}")
    if pooled_summary is None or any(benchmark not in pooled_summary for benchmark in BENCHMARKS):
        raise SystemExit("full results are missing pooled max-t baselines")

    config = {
        "alpha": args.alpha,
        "seed": args.seed,
        "rank_simulations": args.rank_simulations,
        "global_null_repetitions": args.global_null_repetitions,
        "pair_null_repetitions": args.pair_null_repetitions,
        "self_repetitions": args.self_repetitions,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "transfer_bootstrap_repetitions": args.transfer_bootstrap_repetitions,
        "simulation_items": args.simulation_items,
    }
    payload = {
        "config": config,
        "known_rank_simulation": known_rank_coverage(args),
        "global_permutation_null": global_permutation_null(args),
        "pairwise_swap_null": pairwise_swap_null(args, full_summary),
        "heldout_transfer": heldout_transfer(args),
        "cross_harness_self_pair": cross_harness_self_pair(args, full_summary, pooled_summary),
    }
    (results_dir / "extended_validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (results_dir / "EXTENDED_VALIDATION.md").write_text(render_report(payload), encoding="utf-8")
    print(f"wrote {results_dir / 'EXTENDED_VALIDATION.md'}", flush=True)


if __name__ == "__main__":
    main()
