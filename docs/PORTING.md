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
| lpm_rpe1PertEmb | – | r | ported | its Replogle RPE1 input has no pinned md5 yet (not in `fetch-all`) |
| transfer | – | r | ported | Replogle K562 side-load |
| gears | yes | gears-gpu | done | on the dashboard |
| cpa | yes | cpa-gpu | done | norman_from_scfoundation only; on the dashboard |
| **geneformer** | yes | geneformer-gpu / geneformer-blackwell | done | adamson matches paper per perturbation (r ≥ 0.999); in omni-scfm-results |
| **uce**, **uce33** | yes | uce-gpu / uce-blackwell | done | adamson matches paper per perturbation (\|Δ\| < 5e-4); paper's single "UCE*" can't be attributed to 4 vs 33 layers (they differ ≤ 0.005); in omni-scfm-results |
| scfoundation | yes | scfoundation-gpu | ported | runs at batch 4 (paper: 6, needs >24 GB); paper run ≈ 70 h/seed; disabled in the plan |
| scgpt | yes | scgpt-gpu | ported | flash-attn 1.0.4 / torch 1.13 cu117: no Blackwell support → needs a pre-Blackwell GPU |
| scbert | yes | – | todo | paper: RTX 3090, 4–11 h/job |
| lpm_scgptGeneEmb, lpm_scFoundationGeneEmb | – | r | done | adamson matches paper per perturbation (\|Δ\| < 3e-6, Panel C); embeddings are CPU weight lookups (`make-gene-emb-*`), no GPU needed |

## Open issues (not per-method)

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
