# omni-scfm

Omnibenchmark port of "Benchmarking deep learning models for perturbation prediction".

## Overview

This benchmark was created using [OmniBenchmark](https://omnibenchmark.org) v0.5.1, a framework for automated scientific benchmarking.

## Methods

Port of the paper's prediction methods, each run as an OmniBenchmark method module
that executes the corresponding vendored script (`vendor/paper/benchmark/src/`)
verbatim. Status legend: ✅ wired & validated · 🎯 next-up target · ⬜ planned.

**Near-term target ("set 1"): the 4 deep/foundation models + 3 baselines = 7 methods.**

| method | id | type | vendor script | env | status | notes |
|---|---|---|---|---|---|---|
| Mean | `mean` | baseline | `run_mean_prediction.R` | r | ✅ | via `run_r_method.sh` |
| Additive | `additive` | baseline (double-pert) | `run_additive_model.py` | gears | ✅ | bit-exact vs published |
| Linear pretrained (self-trained) | `lpm_selftrained` | baseline (linear) | `run_linear_pretrained_model.R` | r | ✅ | all-NA on double-perts (model limitation) |
| GEARS | `gears` | deep (GNN) | `run_gears.py` | gears | 🎯 | env largely present; trains per split (GPU) |
| scGPT | `scgpt` | foundation | `run_scgpt.py` | new | 🎯 | needs pretrained weights + GPU env |
| Geneformer | `geneformer` | foundation | `run_geneformer.py` | new | 🎯 | needs weights + GPU env |
| scFoundation | `scfoundation` | foundation | `run_scfoundation.py` | new | 🎯 | needs weights + GPU env |
| CPA | `cpa` | deep (autoencoder) | `run_cpa.py` | new | ⬜ | trains per split |
| scBERT | `scbert` | foundation | `run_scbert.py` | new | ⬜ | needs weights + GPU env |
| UCE | `uce` | foundation | `run_uce.py` | new | ⬜ | needs weights + GPU env |
| Transfer (linear) | `transfer` | baseline (cross-dataset) | `run_transfer_perturbation_prediction.R` | r | ⬜ | needs a `--reference_data` dataset (e.g. Replogle) staged as a second input |

Datasets currently wired: `adamson`, `norman`, `norman_from_scfoundation`.

## Author

- **btraven** (ben.uzh@proton.me)

## License

This project is licensed under the MIT License.

## Getting Started

1. Install OmniBenchmark:
   ```bash
   pip install omnibenchmark
   ```

2. Run the benchmark:
   ```bash
   ob run benchmark.yaml
   ```

## Citation

If you use this benchmark in your research, please cite it using the information in `CITATION.cff`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
