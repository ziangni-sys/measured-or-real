# Computing Environment

## Recorded analysis runs

The full robustness run in
`results/disatt_robustness_full_4000x500_seed20270720/` recorded:

- Python 3.13.5
- Windows 11, build 26200
- AMD64 Family 25 Model 117 processor
- 310.7 seconds wall time for the 4000 x 500 robustness experiment

The stored Algorithm 2 summary (`results/algo2_reported/summary.json`) records
2.09 seconds for its main analysis. The statistical analyses need only the
environment pinned in `uv.lock`.

## Machine used to check this repository

- CPU: AMD Ryzen 7 H 260 with Radeon 780M Graphics, 8 cores / 16 logical
  processors
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB
- RAM: 16 GB
- OS: Microsoft Windows 11 Home, 64-bit, version 10.0.26200, build 26200
- Python: 3.13.5
- NumPy: 2.5.1
- SciPy: 1.18.0
- Matplotlib: 3.11.1
- PyArrow: 25.0.0
- pandas: 3.1.0 and PyMuPDF: 1.28.2 (figure scripts only)
- Rust: not on `PATH`, so the Rust commands were not run on this machine

## GPU regeneration

The GPU packages used for the LLM evaluations are pinned in
`REQUIREMENTS-GPU.txt`. The compact correctness and score tensors that the
statistical analyses need are included, so the main checks do not need a GPU.
The exact GPU hardware of every original model-scoring run was not recorded;
it is stated here as unknown rather than inferred from the machine above.
