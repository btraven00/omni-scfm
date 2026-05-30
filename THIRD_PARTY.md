# Third-party code

This benchmark reuses external code. Each entry notes what, where, and under
which license.

## GEARS — vendored `DataSplitter`

- **What:** the `DataSplitter` class and the `parse_single_pert` /
  `parse_combo_pert` / `parse_any_pert` helpers it depends on.
- **Where (here):** `src/omni_scfm/_vendor/gears_split.py` — copied **verbatim**
  from GEARS `gears/data_utils.py` and `gears/utils.py` (only the torch-importing
  helpers `DataSplitter` does not use were omitted).
- **Why:** to reproduce the paper's exact train/val/test split
  (`prepare_split(split='simulation', seed=…)`) without importing the full
  `gears` package, which pulls in `torch` and `torch_geometric`.
- **Source:** https://github.com/snap-stanford/GEARS
- **License:** MIT — Copyright (c) the GEARS authors (Stanford SNAP).
- **Citing:** Roohani, Y., Huang, K. & Leskovec, J. *Predicting transcriptional
  outcomes of novel multigene perturbations with GEARS.* Nat Biotechnol (2023).

## Reference paper code (git submodule)

- **What:** baseline/method scripts reused verbatim where possible.
- **Where (here):** `vendor/paper/` (git submodule, pinned).
- **Source:** https://github.com/const-ae/linear_perturbation_prediction-Paper
- **License:** MIT — Copyright (c) 2024 Constantin Ahlmann-Eltze.
- See also `CITATION.cff` (`references:`).

## GEARS

- **What:** `cell-gears` provides the GO-filtered simulation split (run via the
  vendored `prepare_perturbation_data.py`) and, later, the GEARS method.
- **Where:** `gears` conda env (`envs/gears.yml`), pinned to the paper's versions.
- **Source:** https://github.com/snap-stanford/GEARS — MIT.

## picklerick (SCX)

- **What:** native Rust→R reader for `.h5ad`; the R method scripts read expression
  via `picklerick::read_h5ad` instead of zellkonverter/basilisk (no Python env).
  `wrapper.R` swaps `zellkonverter::readH5AD` to it. Validated: results are
  identical to the basilisk path to ~1e-6.
- **Where:** `r-picklerick` from the `https://prefix.dev/edge` channel (`envs/r.yml`).
- **Source:** https://github.com/btraven00/scx — GPL-3.0-only.
