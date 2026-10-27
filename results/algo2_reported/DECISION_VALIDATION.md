# Fixed-Grid Decision Validation

The simulation compares typed model-comparison decisions when the item-population truth is known. It conditions on a fixed configuration grid and does not assess transfer to unseen prompt populations.

Settings: 1000 families per scenario, 6 models, 2365 paired items, alpha=0.050, and 2000 Gaussian max-t draws.

| Scenario | Method | Exact typed verdict | Stable detection | Reversal detection | False stable | Refusal | Null-family FWER |
|---|---|---:|---:|---:|---:|---:|---:|
| global-null | pooled-max-t | 99.6% | -- | -- | 0.4% | 99.6% | 4.3% |
| global-null | protocol-aware-max-t | 99.6% | -- | -- | 0.0% | 100.0% | 5.1% |
| stable | pooled-max-t | 100.0% | 100.0% | -- | 0.0% | 0.0% | -- |
| stable | protocol-aware-max-t | 89.5% | 89.5% | -- | 0.0% | 10.5% | -- |
| protocol-reversal | pooled-max-t | 0.0% | -- | 0.0% | 64.6% | 35.4% | -- |
| protocol-reversal | protocol-aware-max-t | 68.3% | -- | 68.3% | 0.0% | 31.7% | -- |
| near-tie-stable | pooled-max-t | 33.3% | 33.3% | -- | 0.0% | 66.7% | -- |
| near-tie-stable | protocol-aware-max-t | 4.5% | 4.5% | -- | 0.0% | 95.5% | -- |

`Exact typed verdict` requires the correct direction and class (`stable`, `protocol-dependent`, or `unresolved`). A refusal is not counted as a false stable winner. Pooled max-t controls simultaneous error for its averaged estimand but cannot emit a protocol-dependent verdict, so its behavior in the reversal scenario quantifies the information lost by pooling.

These simulations establish behavior for the stated synthetic item model only. They complement, rather than replace, the real-data permutation, held-out-configuration, and cross-harness diagnostics.
