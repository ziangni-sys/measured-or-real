# Reproducing the Results

Run every command from the repository root after `uv sync`. The Rust commands
(`cargo run --release -p erel-cli -- ...`) need a Rust toolchain. On PowerShell,
write `DATASET=cifar10 uv run ...` as
`$env:DATASET="cifar10"; uv run ...; Remove-Item Env:DATASET`, so that later
commands run on CIFAR-100 again.
Outputs under `results/*_reproduced/` and `results/algo2_alpha*/` are
git-ignored, so a rerun does not overwrite the stored results.

## Algorithm 1 and the disattenuation simulations

- Simulation MSE, Table 3 and Table D.1. Generates its own data.
  ```text
  cargo run --release -p erel-cli -- disatt2
  ```
- Gate ablation, Table D.3. Generates its own data.
  ```text
  cargo run --release -p erel-cli -- ablate
  ```
- Robustness sweep, Figure 3 (a, b), Table D.4 and
  Figure D.1. The stored run is
  `results/disatt_robustness_full_4000x500_seed20270720/`.
  ```text
  uv run python scripts/disatt_robustness.py --repetitions 4000 --bootstrap-repetitions 500 --seed 20270720 --output-dir results/disatt_robustness_reproduced
  uv run python figscripts/fig1_teaser.py
  uv run python figscripts/figD1_robustness.py
  ```
- Simulation check of Proposition 2 (Appendix D.7). Generates its own data.
  ```text
  uv run python scripts/prop2_check.py --repetitions 20000 --large-repetitions 200000 --seed 0
  ```
- Fieller interval and REML closed form (Appendix D.7). Generates its own
  data.
  ```text
  uv run python scripts/interval_compare.py --repetitions 4000 --seed 0
  ```
- Persistence study (Appendix B, Figure B.1). Reads and writes
  `data/synth.csv`.
  ```text
  cargo run --release -p erel-cli -- synth data/synth.csv
  uv run python figscripts/figB1_persistence.py
  ```
- Implementation of Algorithm 1: `crates/erel-core/src/disatt.rs`. Tests:
  `cargo test`.

## CIFAR difficulty scores

The scores of all eight seeds are in `data/prune_scores/` (CIFAR-100) and
`data/prune_scores_cifar10/` (CIFAR-10).

- Seeds 4-7 of the score files. Trains on a GPU and skips a seed whose file
  exists. The environment variables `START`, `END`, `EPOCHS` and `DATASET` set
  the seed range (default 4 to 7), the epochs (default 30) and the dataset.
  ```text
  uv run --with torch --with torchvision --with huggingface-hub --with pillow python scripts/train_seeds.py
  ```
- Score reliabilities and the variance-mean check (Table E.1,
  Appendix E.4, Figure E.1).
  ```text
  uv run python scripts/difficulty_gtheory.py
  DATASET=cifar10 uv run python scripts/difficulty_gtheory.py
  uv run python figscripts/figE1_hetero.py
  ```
- Cross-seed correction (Section 3.5, Table E.2).
  ```text
  uv run python scripts/real_recover.py
  DATASET=cifar10 uv run python scripts/real_recover.py
  ```
- Coverage calibrations (Appendix E.4). Generate their own data.
  ```text
  uv run python scripts/coverage_hetero.py --repetitions 4000 --seed 0
  uv run python scripts/cond_coverage.py --repetitions 20000 --seed 0
  uv run python scripts/boot_calibrate.py --repetitions 1500 --bootstrap-repetitions 400 --seed 0
  ```
- Finite-sample bias of the ICC (Appendix E.6). Generates its own data.
  ```text
  uv run python scripts/icc_bias.py --repetitions 4000 --seed 0
  ```
- Per-class count (Section 3.5, Appendix E.2). Needs the CIFAR-100 labels,
  which are not included (`KNOWN_LIMITATIONS.md`). The first command uses the
  loader of `prune_cifar.py` to download the CIFAR-100 training split and cache
  it as `data/cifar100_hf/train.npz`, in the label order of the scores. It
  trains nothing and needs no GPU.
  ```text
  uv run --with torch --with torchvision --with huggingface-hub --with pillow python -c "import sys; sys.path.insert(0, 'scripts'); import prune_cifar; prune_cifar._load_split('train')"
  uv run python scripts/real_disatt.py
  ```
  Running `scripts/prune_cifar.py` itself also builds this cache, but it then
  retrains nine pruned models on a GPU (keep fractions 0.1, 0.3 and 0.5, each
  with three pruning rules, 40 epochs each) and writes
  `data/prune_cifar.json`.

## Algorithm 2 and the LLM evaluations

The stored output in `results/algo2_reported/` is the one reported in the
paper. Its main summary was computed from the included tensors with base seed
20270734 (each benchmark adds its index to the seed). `extended_validation.json`
and `decision_validation.json` record their own seeds, 20270724 and 20270735.

- Verdicts and the pooled comparison on real data (Table 4,
  Table 5; Appendix G.1, Appendix G.4,
  Appendix G.8; Figure 3 (c), Figure G.4,
  Figure G.1).
  ```text
  uv run python -m scripts.algo2_simrank.run --alpha 0.05 --bootstrap-repetitions 10000 --seed 20270734 --output-dir results/algo2_reproduced
  uv run python figscripts/fig1_teaser.py
  uv run python figscripts/fig2_reversals.py
  uv run python figscripts/figG1_verdictmap.py
  ```
- Held-out wordings and orders, permutation null (Table 4;
  Appendix G.5, Appendix G.6). Run after the previous command.
  ```text
  uv run python -m scripts.algo2_simrank.validate_extended --results-dir results/algo2_reproduced --rank-simulations 200 --global-null-repetitions 200 --pair-null-repetitions 200 --self-repetitions 200 --bootstrap-repetitions 1000 --transfer-bootstrap-repetitions 2000 --seed 20270724
  ```
- Split-grid self-comparison (Table 4, Table 5;
  Appendix G.7; Figure G.3). Stored output only:
  `results/algo2_reported/cross_harness_native.json`. No released script
  writes this file.
  ```text
  uv run python figscripts/figG3_selfcomp.py
  ```
- Simulated benchmarks with known truth (Table 5, Appendix G.3).
  ```text
  uv run python scripts/validate_decisions.py --output-dir results/algo2_reproduced --simulations 1000 --items 2365 --bootstrap-repetitions 2000 --seed 20270735
  ```
- New ARC wordings (Appendix G.6, Figure G.2). Reads
  `data/llm_holdout_*.npy`. By default the summary overwrites
  `data/holdout_wordings.json`; add `--output <file>` to write it elsewhere.
  ```text
  uv run python scripts/holdout_wordings.py --analyze-only --pairwise-path results/algo2_reproduced/pairwise.csv
  uv run python figscripts/figG2_newwordings.py
  ```
- Level sensitivity of Algorithm 2 (Table G.1).
  ```text
  uv run python -m scripts.algo2_simrank.run --alpha 0.01 --bootstrap-repetitions 10000 --seed 20270734 --output-dir results/algo2_alpha001
  uv run python -m scripts.algo2_simrank.run --alpha 0.05 --bootstrap-repetitions 10000 --seed 20270734 --output-dir results/algo2_alpha005
  uv run python -m scripts.algo2_simrank.run --alpha 0.1 --bootstrap-repetitions 10000 --seed 20270734 --output-dir results/algo2_alpha01
  ```
- g(1) and g(32) (Section 4.3, Table F.2, Figure F.1). Reads
  `data/llm_decomp_*.npy`. Run `llm_gtheory.py` once per included tensor, with
  `MODEL` such as `Qwen2.5-0.5B-Instruct` and `BENCHMARK` one of `arc`,
  `hellaswag`, `mmlu`, `openbookqa`.
  ```text
  uv run python scripts/llm_gtheory.py MODEL BENCHMARK
  uv run python scripts/analyze_scale.py 0.5B
  uv run python scripts/analyze_scale.py 1.5B
  uv run python scripts/analyze_scale.py 7B
  uv run python figscripts/figF1_gcoef.py
  ```
- Beta-binomial check (Appendix F.3). Reads `data/llm_decomp_*.npy`.
  ```text
  uv run python scripts/binary_gtheory.py
  ```
- Shot-format control (Figure F.2, Table F.3). Reads
  `data/llm_chatshot_*.npy` and `data/llm_decomp_*.npy`. Without
  `--analyze-only` the script rescores the models on a GPU with the packages
  pinned in `REQUIREMENTS-GPU.txt`.
  ```text
  uv run python scripts/shot_format.py --analyze-only
  uv run python figscripts/fig3_shotformat.py
  ```
- Implementation of Algorithm 2: `scripts/algo2_simrank/core.py`. Tests:
  `uv run python -m unittest discover -s scripts/algo2_simrank/tests -v`.

## Embeddings

- Table H.1, Appendix H. Needs the three feature matrices, which are not
  included. `gauge_probe.py` regenerates them by training the three ResNet-18
  seeds on a GPU; `gauge_probe2.py` then computes the table from
  `data/gauge_feats/`.
  ```text
  uv run --with torch --with torchvision --with huggingface-hub --with pillow python scripts/gauge_probe.py
  uv run python scripts/gauge_probe2.py
  ```
  `gauge_probe.py` also reads `DATASET=cifar10` and then writes to
  `data/gauge_feats_cifar10/`; `gauge_probe2.py` reads `data/gauge_feats/` only.
- Linear-probe check, not reported in the paper. `gauge_constructive.py` reads
  the same three feature matrices and the CIFAR-100 test cache
  `data/cifar100_hf/test.npz` that `gauge_probe.py` writes. On a fixed 80/20
  split of the test images it fits a logistic-regression probe to four
  representations: one seed, the plain mean of the three unaligned seeds, the
  mean after Procrustes alignment, and that mean weighted by per-dimension
  reliability. It prints each probe's accuracy.
  ```text
  uv run --with scikit-learn python scripts/gauge_constructive.py
  ```

## Schematics (Figure 1, Figure 2, Figure 4)

The three schematics are TikZ sources and need a TeX distribution with the
standalone class, TikZ and the newtx fonts. Each run writes the PDF next to
its source; copy it to `figures/` to replace the copy used by the paper.
```text
cd figscripts/schematics
pdflatex schem_overview.tex
pdflatex schem_gate.tex
pdflatex schem_alg2.tex
```
`schem_overview.tex` is Figure 1, `schem_gate.tex` Figure 2 and
`schem_alg2.tex` Figure 4. The interval positions drawn in Figure 1 (b) and
Figure 4 (b) are written into the sources, with comments naming their rows.
They come from `results/algo2_reported/pairwise.csv`, and the pooled interval
in Figure 1 from `results/algo2_reported/pooled_pairwise.csv`.

## Other commands

- Rust sweep over reliability and the number of replicates, with an
  alpha-sensitivity printout: `cargo run --release -p erel-cli -- disatt`.
- Exact-F ICC intervals: `uv run python scripts/icc_ci.py`. Its embedding part
  needs the same feature matrices as Table H.1 and is skipped without them.

## Checks

- `uv run python scripts/verify_release.py` checks the data, the stored results
  and the hashes in `MANIFEST.json`.
- `uv run python scripts/smoke_test.py` runs shortened versions of the CPU
  commands above and every figure script. It writes its outputs and report to
  `results/_smoke/`. Without `cargo` on PATH, the Rust check is reported as an
  expected limitation, not skipped.
