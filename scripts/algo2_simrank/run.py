from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .core import analyze_pooled_scores, analyze_scores
from .data import load_benchmark


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("arc", "hellaswag", "mmlu", "openbookqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simultaneous paired model ranking with protocol robustness")
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20270722)
    parser.add_argument("--reference-weight", type=float, default=0.5)
    parser.add_argument("--weight-min", type=float, default=0.0)
    parser.add_argument("--weight-max", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "algo2_reproduced")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(config: dict, summaries: dict[str, dict], elapsed: float) -> str:
    lines = [
        "# Simultaneous Paired Ranking",
        "",
        "This run implements the fixed-grid simultaneous paired comparison used by Algorithm 2.",
        "",
        "## Method",
        "",
        "The method averages the fixed 4 x 4 wording/order grid within each protocol, forms paired item-level model differences, and calibrates one Gaussian max-t critical value over every model pair and protocol. The resulting simultaneous pairwise intervals are converted into simultaneous rank confidence sets. For two protocols, it also reports the point-estimate flip weight and the largest symmetric interval around the reference weight on which the confidence-supported winner is unchanged.",
        "",
        "The simultaneous-ranking construction is adapted to paired benchmark correctness from *Prompt-Dependent Ranking of Large Language Models with Uncertainty Quantification* (Avelar Menendez, Liu, and Dai, 2026, arXiv:2603.03336). It does not use their contextual BTL likelihood because these data are paired correctness measurements rather than human pairwise preferences.",
        "",
        "## Scope",
        "",
        "Coverage is asymptotic over benchmark items and conditional on the observed uniform configuration grid. It does not cover unseen wording/order populations, alternative harnesses, or the choice of protocol weights. Weight robustness is a sensitivity certificate over the requested interval, not a probability distribution over protocols.",
        "",
        "## Run",
        "",
        f"Runtime: {elapsed:.2f} seconds.",
        "",
        "| Benchmark | Models | Pairs | Max-t critical value | Stable | Protocol-dependent | Unresolved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark, summary in summaries.items():
        counts = summary["verdict_counts"]
        stable = counts.get("stable-a", 0) + counts.get("stable-b", 0)
        unresolved = counts.get("unresolved", 0) + counts.get("unresolved-mixed", 0)
        lines.append(
            f"| {benchmark} | {summary['n_models']} | {summary['n_pairs']} | {summary['critical_value']:.3f} | "
            f"{stable} | {counts.get('protocol-dependent', 0)} | {unresolved} |"
        )
    lines.extend(
        [
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(config, indent=2),
            "```",
            "",
            "Machine-readable pairwise intervals are in `pairwise.csv`; simultaneous rank sets are in `ranks.csv`; complete metadata are in `summary.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {output_dir}; pass --overwrite to replace named outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    all_pairs: list[dict] = []
    all_ranks: list[dict] = []
    pooled_pairs: list[dict] = []
    pooled_ranks: list[dict] = []
    summaries: dict[str, dict] = {}
    pooled_summaries: dict[str, dict] = {}
    skipped: dict[str, list[dict[str, str]]] = {}
    for benchmark_index, benchmark in enumerate(args.benchmarks):
        model_names, tensors, skipped_models = load_benchmark(ROOT / "data", benchmark)
        # The estimand is the uniform mean over the observed wording/order grid.
        scores = tensors.mean(axis=(1, 2))
        result = analyze_scores(
            scores,
            model_names,
            alpha=args.alpha,
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed + benchmark_index,
            reference_weight=args.reference_weight,
            weight_range=(args.weight_min, args.weight_max),
            batch_size=args.batch_size,
        )
        pooled_result = analyze_pooled_scores(
            scores,
            model_names,
            alpha=args.alpha,
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed + 10_000 + benchmark_index,
            protocol_weight=args.reference_weight,
            batch_size=args.batch_size,
        )
        pairs = result.pop("pairs")
        ranks = result.pop("ranks")
        for row in pairs:
            row = {"benchmark": benchmark, **row}
            all_pairs.append(row)
        for row in ranks:
            row = {"benchmark": benchmark, **row}
            all_ranks.append(row)
        result["models"] = model_names
        summaries[benchmark] = result
        current_pooled_pairs = pooled_result.pop("pairs")
        current_pooled_ranks = pooled_result.pop("ranks")
        pooled_pairs.extend({"benchmark": benchmark, **row} for row in current_pooled_pairs)
        pooled_ranks.extend({"benchmark": benchmark, **row} for row in current_pooled_ranks)
        pooled_summaries[benchmark] = pooled_result
        skipped[benchmark] = skipped_models
        print(
            f"{benchmark}: {result['n_models']} models, {result['n_pairs']} pairs, "
            f"q={result['critical_value']:.3f}, verdicts={result['verdict_counts']}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    config = {
        "benchmarks": args.benchmarks,
        "alpha": args.alpha,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "reference_weight": args.reference_weight,
        "weight_range": [args.weight_min, args.weight_max],
        "batch_size": args.batch_size,
        "input_target": "uniform fixed 4x4 wording/order grid within each protocol",
        "protocol_weight_definition": "w * zero-shot + (1-w) * few-shot",
    }
    write_csv(output_dir / "pairwise.csv", all_pairs)
    write_csv(output_dir / "ranks.csv", all_ranks)
    write_csv(output_dir / "pooled_pairwise.csv", pooled_pairs)
    write_csv(output_dir / "pooled_ranks.csv", pooled_ranks)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "runtime_seconds": elapsed,
                "benchmarks": summaries,
                "pooled_baselines": pooled_summaries,
                "skipped": skipped,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(build_report(config, summaries, elapsed), encoding="utf-8")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
