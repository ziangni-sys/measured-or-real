# Measured or Real? When Model-Derived Measurements Can Be Corrected or Compared

Code, data and prompt texts for the paper by

- **Yifeng Li**, University of Trento (yifeng.li@studenti.unitn.it)
- **Ziang Ni**, Delft University of Technology (ziangni@tudelft.nl)

The two authors contributed equally.

## What the paper does

Numbers computed from trained models, such as an example's difficulty score or
a model's lead on a benchmark, change with the training seed and the prompt.
The paper asks when such numbers can be corrected or compared, in three parts.

1. **Correcting correlations for replicate noise (Algorithm 1).** Noise across
   seeds shrinks the correlation between two scores. The textbook correction
   divides by the square root of two estimated reliabilities and can return
   values above one. Algorithm 1 corrects only when both reliabilities pass an
   F-test, then divides by their upper confidence bounds, so its output stays
   bounded. The Rust crate in `crates/` implements it and runs the simulations.
2. **Comparing models across prompt protocols (Algorithm 2).** Prompt settings,
   such as zero-shot versus two-shot, can reverse a ranking. Algorithm 2 builds
   simultaneous intervals for every model pair within each protocol and names a
   winner only when every protocol agrees. Otherwise the pair is
   protocol-dependent or refused. The code is in `scripts/algo2_simrank/`.
3. **Aligning embedding coordinates.** Per-dimension reliability of embeddings
   across seeds is near zero because the coordinates are arbitrary. After an
   orthogonal Procrustes alignment fitted on held-out images, the same features
   are reliable.

## Repository layout

```text
measured-or-real/
├── crates/
│   ├── erel-core/        Algorithm 1, reliability estimators, simulation designs (Rust)
│   └── erel-cli/         `erel` command: disatt2, ablate, synth, disatt
├── scripts/
│   ├── algo2_simrank/    Algorithm 2, its validations and unit tests
│   ├── *.py              analyses of scores, LLM tensors, embeddings and coverage
│   ├── verify_release.py integrity and hash checks
│   ├── build_manifest.py writes MANIFEST.json
│   └── smoke_test.py     shortened runs of the CPU entry points
├── figscripts/           one plotting script per data figure, plus style.py
│   └── schematics/       TikZ sources of the three schematics
├── figures/              the paper's figures (PDF)
├── data/                 LLM correctness tensors, CIFAR difficulty scores, simulation output
├── results/              stored outputs of Algorithm 2 and of the robustness sweep
├── REPRODUCIBILITY.md    result-to-command map with all flags and seeds
├── PROMPTS.md            exact prompt texts
├── ENVIRONMENT.md        recorded software and hardware
├── KNOWN_LIMITATIONS.md  inputs that must be regenerated
├── MANIFEST.json         SHA-256 of every released file
├── pyproject.toml, uv.lock, .python-version
└── Cargo.toml, Cargo.lock, justfile
```

## Quick start

The Python analyses need [uv](https://docs.astral.sh/uv/) and run on a CPU.

```text
uv sync
uv run python scripts/verify_release.py
uv run python -m unittest discover -s scripts/algo2_simrank/tests -v
uv run python scripts/smoke_test.py
```

`verify_release.py` checks the data shapes, the stored results and the file
hashes in `MANIFEST.json`. `smoke_test.py` runs shortened versions of the CPU
commands and the figure scripts, and writes its report to
`results/_smoke/SMOKE_TEST_REPORT.md`. After changing files, refresh the
manifest with `uv run python scripts/build_manifest.py`.

## Rust

The simulations of Algorithm 1 need a Rust toolchain.

```text
cargo build --release
cargo test
cargo run --release -p erel-cli -- disatt2
cargo run --release -p erel-cli -- ablate
```

The binary is `target/release/erel` (`target\release\erel.exe` on Windows).
A `justfile` wraps the common commands.

## Figures

`figures/` holds every figure of the paper as a PDF. The table names the
source of each one. File names are working names and do not follow the
paper's figure numbers.

| Figure | Source |
| --- | --- |
| Figure 1 | `figscripts/schematics/schem_overview.tex` |
| Figure 2 | `figscripts/schematics/schem_gate.tex` |
| Figure 3 | `figscripts/fig1_teaser.py` |
| Figure 4 | `figscripts/schematics/schem_alg2.tex` |
| Figure B.1 | `figscripts/figB1_persistence.py` |
| Figure D.1 | `figscripts/figD1_robustness.py` |
| Figure E.1 | `figscripts/figE1_hetero.py` |
| Figure F.1 | `figscripts/figF1_gcoef.py` |
| Figure F.2 | `figscripts/fig3_shotformat.py` |
| Figure G.1 | `figscripts/figG1_verdictmap.py` |
| Figure G.2 | `figscripts/figG2_newwordings.py` |
| Figure G.3 | `figscripts/figG3_selfcomp.py` |
| Figure G.4 | `figscripts/fig2_reversals.py` |

Data figures come from the Python scripts `figscripts/*.py`. Run one from the
repository root (from another directory, add `--project <repo>` and give the
script's full path):

```text
uv run python figscripts/fig1_teaser.py
```

The PDF goes to `figures/` and a PNG preview to `figures/preview/`. The
variables `FIG_REPO`, `FIG_OUT` and `FIG_PREVIEW` change these locations. Each
script checks the values it plots and stops if a check fails.

The three schematics come from the TikZ sources `figscripts/schematics/*.tex`.
Run pdflatex on a file in that folder and copy the PDF to `figures/`. The
README in that folder says what each file draws and what it needs.

## Known limitations

- Two inputs are not included and must be regenerated on a GPU: the CIFAR-100
  class labels (for the per-class count) and the three embedding feature
  matrices (for the embedding table). See `KNOWN_LIMITATIONS.md`.
- The exact revisions of the LLM checkpoints were not recorded. The included
  correctness tensors are the reproducibility target for every LLM result.
- The split-grid self-comparison counts are a stored output with no script that
  regenerates them, and no script computes the joint cross-seed estimate.
- The Rust code implements only the exact-F branch of Algorithm 1.

## Citation

```bibtex
@misc{li2026measured,
  title  = {Measured or Real? When Model-Derived Measurements Can Be Corrected or Compared},
  author = {Li, Yifeng and Ni, Ziang},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

MIT. See `LICENSE`.
