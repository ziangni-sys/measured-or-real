# Schematic sources

TikZ sources of the three schematic figures. The data figures come from the
Python scripts one folder up.

| File | Figure | What it draws |
| --- | --- | --- |
| `schem_overview.tex` | Figure 1 | Overview at full text width. Left: one measurement model and the rule that makes each facet a replicate or a protocol. Right: three lanes, (a) correct a correlation, (b) compare models, (c) measure embedding reliability. |
| `schem_gate.tex` | Figure 2 | One column. The factor that one estimated reliability adds to the correction, for the plug-in correction and for Algorithm 1, with the gate and the declined region. |
| `schem_alg2.tex` | Figure 4 | One column. (a) How Algorithm 2 averages over configurations, keeps the protocols apart and builds one interval per protocol. (b) Four ARC pairs and their verdicts. |

Each file is drawn at its final size, and the paper includes the PDF at 100%.
The macros and the palette that a file uses are copied into it, so each file
builds on its own.

## Build

You need a TeX distribution with the standalone class, TikZ and the newtx fonts,
such as TeX Live or MiKTeX. Run pdflatex in this folder, once per file:

```text
cd figscripts/schematics
pdflatex schem_overview.tex
pdflatex schem_gate.tex
pdflatex schem_alg2.tex
```

Each run writes `<file>.pdf` next to its source. Copy the PDF to `figures/` to
replace the copy used by the paper. The `.aux`, `.log` and `.pdf` files in this
folder are build output, and git ignores them.
