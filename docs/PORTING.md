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
| scfoundation | yes | scfoundation-gpu / scfoundation-blackwell | ported | paper settings now: batch 6 + **epochs 5** (the paper's submission, not the script default 15); one job fills ~94 GB, 11,399 steps/epoch at ~2.5 s/step ≈ 39 h/seed (paper: 70 h), so seeds run one after the other; norman_from_scfoundation seeds 1,2 running on the Blackwell box |
| scgpt | yes | scgpt-gpu | ported | flash-attn 1.0.4 / torch 1.13 cu117: no Blackwell support → needs a pre-Blackwell GPU |
| scbert | yes | scbert-gpu / scbert-blackwell | done | adamson seed 1 (paper env, roland L4, 100 epochs, 13 h) matches the paper per perturbation to float32 rounding (\|Δ pearson_delta\| ≤ 2e-6, all 24 paper rows), which also vouches for the re-uploaded checkpoint; norman_from_scfoundation + seed 2 still to run (box) |
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
- **Replogle K562 / RPE1 as benchmark datasets:** the paper runs every method on them
  (its longest jobs, e.g. Geneformer ≈ 11.5 h each); here they are only side-loaded
  references for `transfer` / the lpm embeddings. Scope decision.
- **norman_from_scfoundation split ≠ paper's split:** `scf_split` seeds numpy before
  GEARS loads the data, whose RNG use varies by setup, so test sets overlap the paper's
  only partly. Split-independent methods (additive) still match; split-dependent ones
  (geneformer, gears, …) are comparable within this benchmark, not per perturbation to
  the paper.
- **Blackwell (sm_120) GPUs:** every paper env predates them; the `*-blackwell` envs bump
  torch only (validated numerically harmless for geneformer on adamson).

Side-load weights/data for all of the above: `pixi run fetch-all` (see AGENTS.md).
