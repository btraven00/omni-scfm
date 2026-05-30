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

## Reference paper code (planned: git submodule)

- **What:** baseline/method scripts reused verbatim where possible.
- **Where (here):** `vendor/paper/` (git submodule, to be added).
- **Source:** https://github.com/const-ae/linear_perturbation_prediction-Paper
- **License:** MIT — Copyright (c) 2024 Constantin Ahlmann-Eltze.
- See also `CITATION.cff` (`references:`).
