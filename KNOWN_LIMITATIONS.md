# Known Limitations

## Inputs that are not included

Two inputs used by analysis scripts are not in this repository and could not
be recovered:

1. `data/gauge_feats/feat_seed{0,1,2}_e20.npy`, the three CIFAR-100 test-set
   feature matrices behind the embedding table.
2. `data/cifar100_labels.npy` (or the CIFAR-100 cache `train.npz` written by
   `scripts/prune_cifar.py`), used only to split the included 50,000
   per-example scores by CIFAR-100 class in `scripts/real_disatt.py`.

The embedding table and the per-class count therefore cannot be recomputed
from this repository alone. Their source code is included.
`scripts/gauge_probe.py` regenerates the feature matrices by training the three
ResNet-18 seeds after downloading CIFAR-100, and `scripts/prune_cifar.py`
downloads the public CIFAR-100 parquet and recreates its local cache. No
replacement numbers or synthetic stand-ins were inserted.

The included score arrays reproduce the pooled cross-seed difficulty analyses
without the labels. The LLM correctness tensors, all Algorithm 2 analyses, the
disattenuation simulations, the robustness experiment and the prompt-control
analyses run from the included files.

## LLM checkpoints and GPU runs

The exact model revisions and the exact GPU hardware of every LLM scoring job
were not recorded. The included correctness tensors are therefore the
reproducibility target for the statistical claims.

`scripts/llm_decomp.py`, which scores the models on a GPU, imports a module
`scripts/algo2_input_provenance.py` that is not included. The GPU paths of
`scripts/shot_format.py` and `scripts/holdout_wordings.py` import
`llm_decomp.py`. Their `--analyze-only` paths, which read the included tensors,
do not.

## Outputs without a regenerating script

- `results/algo2_reported/cross_harness_native.json`, the split-grid
  self-comparison counts, is a stored output; no released script writes it.
- No released script computes the joint cross-seed estimate of the pooled
  CIFAR analysis.
- The self-pair section of `results/algo2_reported/EXTENDED_VALIDATION.md`
  and `extended_validation.json` uses its own critical values (for example
  3.541 for ARC, where `summary.json` has 3.513) and has no pooled column. The
  self-pair counts in the paper are those in `cross_harness_native.json`, which
  uses the critical values in `summary.json`.

## Parts of Algorithm 1 not in the code

Every reported number for Algorithm 1 uses the exact-F branch, and the code
implements only that branch. The algorithm's in-line variance-mean check (the
Spearman correlation that switches to the max rule), the unit-bootstrap bound
of the max rule and the Spearman-Brown map to a different number of replicates
are not part of that implementation. The stand-alone variance-mean diagnostics
of the CIFAR scores (`scripts/difficulty_gtheory.py`,
`figscripts/figE1_hetero.py`) and the bootstrap coverage calibration
(`scripts/boot_calibrate.py`) are separate scripts.
