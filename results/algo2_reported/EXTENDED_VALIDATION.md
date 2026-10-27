# Extended Validation of Simultaneous Paired Ranking

## Known-rank simulation

Across 200 independently generated benchmark families, all protocol-specific true ranks were simultaneously contained in their reported rank sets in 200 runs (1.000, MC SE 0.000) at nominal alpha=0.050. Marginal model-by-protocol rank coverage was 1.000; mean rank-set width was 0.84 positions.

## Global model-label permutation null

Each item keeps its complete two-protocol model profile, but profiles are randomly reassigned to model labels. Any resolved pair/protocol ordering is false under this global null.

| Benchmark | False families | Empirical FWER | MC SE | Resolved pair-replicates |
|---|---:|---:|---:|---:|
| arc | 13/200 | 0.065 | 0.017 | 15/21000 |
| hellaswag | 12/200 | 0.060 | 0.017 | 14/11000 |
| mmlu | 14/200 | 0.070 | 0.018 | 17/11000 |
| openbookqa | 10/200 | 0.050 | 0.015 | 13/11000 |

## Held-out configuration transfer

Rank sets are fit on two of four wording/order levels and checked against ranks computed from the held-out two levels. These are transfer diagnostics, not covered by the item-sampling alpha guarantee.

| Benchmark/axis | Stable direction failures | Protocol rank coverage | Protocol family coverage | Robust endpoint coverage |
|---|---:|---:|---:|---:|
| arc/wording | 11/555 | 148/180 | 0/6 | 168/180 |
| arc/order | 0/553 | 180/180 | 6/6 | 180/180 |
| hellaswag/wording | 4/257 | 122/132 | 2/6 | 128/132 |
| hellaswag/order | 0/248 | 130/132 | 5/6 | 132/132 |
| mmlu/wording | 6/251 | 115/132 | 1/6 | 123/132 |
| mmlu/order | 1/246 | 130/132 | 5/6 | 130/132 |
| openbookqa/wording | 13/261 | 101/132 | 0/6 | 118/132 |
| openbookqa/order | 0/258 | 132/132 | 6/6 | 132/132 |

## Cross-harness self-pairs

Each model is compared with itself using disjoint random halves of the 16 configuration cells. A stable winner is necessarily a harness artifact.

| Benchmark | Stable false winners | Any resolved comparison | Total |
|---|---:|---:|---:|
| arc | 153 | 664 | 3000 |
| hellaswag | 34 | 441 | 2200 |
| mmlu | 31 | 304 | 2200 |
| openbookqa | 118 | 780 | 2200 |

## Pairwise swap diagnostic

This item-wise pair swap uses the deployed max-t critical value. It is a per-pair diagnostic, not a coherent global-null FWER experiment.

| Benchmark | Stable false winners | Protocol-dependent | Pair-replicates |
|---|---:|---:|---:|
| arc | 2 | 0 | 21000 |
| hellaswag | 2 | 0 | 11000 |
| mmlu | 2 | 0 | 11000 |
| openbookqa | 1 | 0 | 11000 |

## Interpretation boundary

The known-rank and permutation experiments assess item-sampling calibration for the fixed-grid target. Held-out and self-pair checks measure sensitivity to the small observed configuration design. Their failures block any claim of coverage for an unrestricted future prompt population; they do not invalidate inference conditional on the prespecified shared grid.
