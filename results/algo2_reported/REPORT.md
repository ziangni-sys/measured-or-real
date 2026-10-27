# Simultaneous Paired Ranking

This run implements the fixed-grid simultaneous paired comparison used by Algorithm 2.

## Method

The method averages the fixed 4 x 4 wording/order grid within each protocol, forms paired item-level model differences, and calibrates one Gaussian max-t critical value over every model pair and protocol. The resulting simultaneous pairwise intervals are converted into simultaneous rank confidence sets. For two protocols, it also reports the point-estimate flip weight and the largest symmetric interval around the reference weight on which the confidence-supported winner is unchanged.

The simultaneous-ranking construction is adapted to paired benchmark correctness from *Prompt-Dependent Ranking of Large Language Models with Uncertainty Quantification* (Avelar Menendez, Liu, and Dai, 2026, arXiv:2603.03336). It does not use their contextual BTL likelihood because these data are paired correctness measurements rather than human pairwise preferences.

## Scope

Coverage is asymptotic over benchmark items and conditional on the observed uniform configuration grid. It does not cover unseen wording/order populations, alternative harnesses, or the choice of protocol weights. Weight robustness is a sensitivity certificate over the requested interval, not a probability distribution over protocols.

## Run

Runtime: 2.09 seconds.

| Benchmark | Models | Pairs | Max-t critical value | Stable | Protocol-dependent | Unresolved |
|---|---:|---:|---:|---:|---:|---:|
| arc | 15 | 105 | 3.513 | 93 | 5 | 7 |
| hellaswag | 11 | 55 | 3.336 | 42 | 4 | 9 |
| mmlu | 11 | 55 | 3.344 | 43 | 3 | 9 |
| openbookqa | 11 | 55 | 3.380 | 43 | 4 | 8 |

## Configuration

```json
{
  "benchmarks": [
    "arc",
    "hellaswag",
    "mmlu",
    "openbookqa"
  ],
  "alpha": 0.05,
  "bootstrap_repetitions": 10000,
  "seed": 20270734,
  "reference_weight": 0.5,
  "weight_range": [
    0.0,
    1.0
  ],
  "batch_size": 2048,
  "input_target": "uniform fixed 4x4 wording/order grid within each protocol",
  "protocol_weight_definition": "w * zero-shot + (1-w) * few-shot"
}
```

Machine-readable pairwise intervals are in `pairwise.csv`; simultaneous rank sets are in `ranks.csv`; complete metadata are in `summary.json`.
