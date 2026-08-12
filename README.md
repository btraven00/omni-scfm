# omni-scfm

Omnibenchmark port of:

> Deep learning-based predictions of gene perturbation effects do not yet
> outperform simple linear baselines. Constantin Ahlmann-Eltze, Wolfgang Huber,
> Simon Anders. Nature Methods 2025; doi:
> https://doi.org/10.1038/s41592-025-02772-6
[git repo](https://github.com/const-ae/linear_perturbation_prediction-Paper)

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
| scGPT | `scgpt` | foundation | `run_scgpt.py` | scgpt-gpu | 🎯 | weights via omni-huggingface (`perturblab/scgpt-human`); needs GPU |
| Geneformer | `geneformer` | foundation | `run_geneformer.py` | new | 🎯 | needs weights + GPU env |
| scFoundation | `scfoundation` | foundation | `run_scfoundation.py` | new | 🎯 | needs weights + GPU env |
| CPA | `cpa` | deep (autoencoder) | `run_cpa.py` | cpa-gpu | ✅ | trains per split (GPU); scoped to `norman_from_scfoundation` |
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

2. Fetch the shared reference data (once):
   ```bash
   pixi run fetch-godata
   ```
   Downloads GEARS' `gene2go_all.pkl` (md5-pinned) into `data/godata/`. The `split`
   and the GEARS/CPA/additive method modules read it from there, avoiding a ~9 MB
   re-download on every split/method/seed. It's *side-loaded* rather than an OB stage
   because OB 0.5.1 can't wire one global file into per-dataset lineages — see
   `AGENTS.md`.

3. Run the benchmark:
   ```bash
   ob run benchmark.yaml
   ```

## Citation

If you use this benchmark in your research, please cite it using the information in `CITATION.cff`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
