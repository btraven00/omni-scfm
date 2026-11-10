# Porting checklist

Every method in Ahlmann-Eltze, Huber & Anders (2025) vs. its state in omni-scfm.
Update this table in the same commit that changes a method's status.

Status: **done** = ported + validated against the paper · **ported** = runs under OB,
paper-setting run still pending · **todo** = not ported.

| method | GPU | env | status | notes |
|---|---|---|---|---|
| ground_truth | – | base | done | |
| mean | – | r | done | on the dashboard |
| additive | – | gears | done | bit-exact vs paper on norman_from_scfoundation; on the dashboard |
| lpm_selftrained | – | r | done | on the dashboard; all-NA on double-pert datasets (paper limitation) |
| lpm_randomPertEmb, lpm_randomGeneEmb | – | r | done | |
| lpm_gearsPertEmb, lpm_k562PertEmb | – | r | done | embeddings side-loaded (`make-pert-emb-*`) |
| lpm_rpe1PertEmb | – | r | done | adamson seeds 1–2 match paper per perturbation (\|Δ\| ≤ 1.3e-5, Panel C); RPE1 pinned 2026-10-05 (`fetch-replogle-rpe1`, in `fetch-all`); embedding tracked in `data/embeddings/` |
| transfer | – | r | ported | Replogle K562 side-load |
| gears | yes | gears-gpu | done | on the dashboard |
| cpa | yes | cpa-gpu | done | norman_from_scfoundation only; on the dashboard |
| **geneformer** | yes | geneformer-gpu / geneformer-blackwell | done | adamson matches paper per perturbation (r ≥ 0.999); in omni-scfm-results |
| **uce**, **uce33** | yes | uce-gpu / uce-blackwell | done | adamson matches paper per perturbation (\|Δ\| < 5e-4); paper's single "UCE*" can't be attributed to 4 vs 33 layers (they differ ≤ 0.005); in omni-scfm-results |
| scfoundation | yes | scfoundation-gpu / scfoundation-blackwell | done | paper settings: batch 6 + **epochs 5** (the paper's submission, not the script default 15); one job fills ~94 GB, so seeds run one after the other (~81 h/seed on the RTX PRO 6000, torch MHA fast path off: it segfaults in eval under torch 2.7.1). norman_from_scfoundation seeds 1,2 done (test doubles mean pearson_delta 0.569 / 0.590 vs paper 0.50–0.61; our split ≠ the paper's, so no per-perturbation match); in omni-scfm-results |
| scgpt | yes | scgpt-gpu | done | flash-attn 1.0.4 / torch 1.13 cu117: no Blackwell support → pre-Blackwell GPU. Builds and runs on roland's L4 (full CUDA 11.7.1 toolkit + g++ 11 + setuptools 69.5.1, see `envs/scgpt-gpu.yml`); paper batch 64 trains in 24 GB, the final predict() needs `OMNI_SCGPT_PRED_BATCH=16` (inference only, no numerical effect). norman_from_scfoundation seeds 1,2 done on roland (~7.5 h/seed; shared test perts r 0.99 / 0.97 vs paper per perturbation, our split ≠ the paper's); in omni-scfm-results |
| scbert | yes | scbert-gpu / scbert-blackwell | done | adamson seed 1 (paper env, roland L4, 100 epochs, 13 h) matches the paper per perturbation to float32 rounding (\|Δ pearson_delta\| ≤ 2e-6, all 24 paper rows), which also vouches for the re-uploaded checkpoint; seeds 1,2 on adamson + norman_from_scfoundation done on the box (scbert-blackwell); in omni-scfm-results |
| lpm_scgptGeneEmb, lpm_scFoundationGeneEmb | – | r | done | adamson matches paper per perturbation (\|Δ\| < 3e-6, Panel C); embeddings are CPU weight lookups (`make-gene-emb-*`), no GPU needed |

## Open issues (not per-method)

- **scFoundation's per-epoch train-set evaluation only prints, but costs ~40% of the run
  and consumes RNG.** After every epoch the forked GEARS evaluates the whole train set
  (68k cells on norman_from_scfoundation; >5 h per pass on the RTX PRO 6000) and then val.
  The train-set metrics are only printed (and optionally logged to wandb); best-model
  selection uses val only (`gears.py:410–436`). Dropping it would not change model
  selection, but the train loader is `shuffle=True` on torch's global RNG, so the extra
  pass shifts every later epoch's shuffle order: not bit-identical. Kept as in the paper;
  details in `modules/methods/scfoundation/run.sh`.
- **Full paper grid (decided 2026-11-08: reproduce all of it).** Replogle K562 / RPE1
  are now `download` modules (seeds 1,2, same Dataverse files as the side-load fetcher;
  the fetch path is being redone). Still open before the grid is complete:
  - **scgpt on the singles** (adamson, K562, RPE1): needs the script's else-branch +
    `essential_all_data_pert_genes.pkl` side-load; excluded until ported.
  - **cpa on the singles**: re-enabled in the plan, not yet run.
  - **gears on norman_from_scfoundation**: in the paper's Panel A, but its run_gears.py
    uses the forked GEARS 0.0.2 there; our module dodges the fork, so it's excluded.
  - **transfer** on K562 uses RPE1 as reference (the paper's K562↔RPE1 supplement).
- **norman_from_scfoundation split = the paper's split since 2026-11-10; earlier results
  used a different one.** The paper seeds numpy, loads the data, then draws the split.
  On its cluster the load hit a cell-graph cache and drew nothing; our per-job sandbox
  rebuilds the cache, which draws from np.random and shifted the split (test sets
  overlapped the paper's at chance level). `scf_split.py` now seeds again after the
  load, which reproduces the paper's test and val sets for all five seeds. Every
  split-dependent result on this dataset from before the fix (all methods but additive)
  has to be rerun; until then it is comparable within this benchmark only, not per
  perturbation to the paper.
- **Blackwell (sm_120) GPUs:** every paper env predates them; the `*-blackwell` envs bump
  torch only (validated numerically harmless for geneformer on adamson).

Side-load weights/data for all of the above: `pixi run fetch-all` (see AGENTS.md).
