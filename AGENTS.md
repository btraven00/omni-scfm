# AGENTS.md — working conventions for omni-scfm

Instructions for agents (and humans) working in this repo. Keep it short; when a
rule here conflicts with an ad-hoc request, surface the conflict rather than
silently breaking the convention.

## Environments

**The pixi feature is the source of truth for every environment.** Each env is a
`[feature.<id>]` in `pixi.toml`, wired into `[environments]`, and materialized
locally by `pixi install -e <id>` → `.pixi/envs/<id>/`. The conda YAMLs under
`envs/*.yml` are the OmniBenchmark-facing mirror (OB builds them via
`snakemake --use-conda` on the run box). Some are generated (`pixi run export-*`);
the heavier/pip-coupled ones (`gears`, `gears-gpu`, `cpa-gpu`) are hand-maintained
to mirror their pixi feature — **if you change one, change both.**

- Run pixi with the project manifest, never a stray one:
  `env -u PIXI_PROJECT_MANIFEST pixi run --manifest-path "$PWD/pixi.toml" -e <id> …`
  (a stray `PIXI_PROJECT_MANIFEST` from another checkout hijacks resolution).
- GPU envs (`gears-gpu`, `cpa-gpu`) need `[feature.<id>.system-requirements] cuda`
  for local pixi installs (pixi doesn't auto-detect the driver; OB's conda build does).
- pip-coupled stacks (e.g. `cpa-gpu`): pixi's uv resolver is **stricter than pip**.
  When a package caps a conda-provided build/meta tool (setuptools, packaging, …),
  pin that tool conda-side in `[feature.<id>.dependencies]` so uv can satisfy it.
- **Never put an environment in `scratch/`.** `scratch/` is throwaway, git-ignored,
  and must not be a dependency of anything tracked (tests, modules, docs).

## Testing

Tests live in `tests/`, selected by `pyproject.toml`'s pytest config. Two tiers:

1. **Unit tests** (default): pure-Python, no envs, no network, no GPU — must always
   pass on any machine. Run: `pixi run -e default python -m pytest`.
2. **Integration tests** (`@pytest.mark.integration`): drive a method's `run.sh`
   end-to-end in its real env on the committed `tests/fixtures/norman_tiny/` fixture,
   and assert the output contract (gzipped `{condition: [per-gene value]}` +
   `gene_names.json`, vectors == gene-panel width). Run: `… -m integration`.

Integration-test rules:
- **Skip, don't fail, on a missing prerequisite** (env, gene2go cache, GPU). The
  suite stays green anywhere; it only *guards* where it can actually run.
- **Env discovery order:** an explicit `OMNI_<METHOD>_ENV_BIN` override (the OB run
  box, where the env is a snakemake conda prefix, not pixi) → else `.pixi/envs/<id>`.
  Nothing else — in particular, never a `scratch/` build.
- **GPU methods are GPU-gated, not excluded:** if the train is short on the fixture
  (CPA ≈ 5 s), include it and skip unless a CUDA device is present; if it's heavy
  (GEARS), leave it out and validate separately.
- **Keep tests module-portable.** Parametrize only by method name + `run.sh` path +
  the committed fixture — nothing repo-global. When a module graduates to its own
  git repo (a planned step), the test function + the `norman_tiny/` fixture dir move
  with it unchanged. Don't introduce cross-module or repo-root coupling in a test.

Adding a method = add its `run.sh` + entrypoint + `benchmark.yaml` module + a pixi
feature + (if not exportable) a hand-mirrored `envs/<id>.yml` + one integration test
following the rules above + its row in `docs/PORTING.md` (the per-method checklist).

## Method modules (`run.sh`)

- **One private sandbox per job.** Each method `run.sh` does `wd=$(mktemp -d); trap
  'rm -rf "$wd"' EXIT`, stages a private copy of the GEARS pert-data folder under it,
  and runs the vendored script with `( cd "$wd" && … )`. This is load-bearing for
  **parallel correctness**: the vendored GEARS caches `data_pyg/cell_graphs.pkl` (and
  the co-expression csv) under a path keyed only by `data_path/<dataset>/` — **no split
  or seed**. Run two seeds in the same folder and they overwrite / silently reuse each
  other's cache (and the basal sampling is seed-dependent, so reuse is *wrong*). The
  per-job `$wd` makes the absolute path unique per `(dataset, seed, method)`, so OB's
  concurrent seeds can't collide. Don't "optimize" by sharing a pert-data folder.
- **Side-loaded data lives in the user repo, but OB runs from the staged commit dir.**
  Under `ob run`, a module's cwd is `out/.modules/omni-scfm/<commit>/`, so `$(pwd)` is
  NOT the user repo — and `data/` is git-ignored, so it isn't staged there. Resolve
  side-loaded paths from the **user repo = parent of `out/`**, recovered from
  `--output_dir`: `DATA_ROOT="${output_dir%%/out/*}"` (fall back to `$REPO` for local
  invocation). Use `$DATA_ROOT/data/...` for the checkpoint / gene2go / go_essential
  defaults, never `$REPO/data/...`. (Vendored code under `vendor/` IS committed, so
  `$REPO/vendor/...` is fine.)

## Reference data (side-loading)

Global, dataset-independent reference files (e.g. GEARS' `gene2go_all.pkl`) are
**side-loaded**, not OB stages: OB 0.5.1 can't wire one shared artifact into multiple
per-dataset lineages (a parallel-root input is silently dropped from argv). Pattern:
an md5-pinned fetcher under `modules/<group>/<name>/run.sh` + a `pixi run fetch-*`
task that lands it in a git-ignored `data/<group>/` dir; consuming `run.sh` scripts
**default to that conventional path** (with a flag override). Run the fetch once
before `ob run`. Don't try to express it as an OB `inputs:` node until OB grows a
first-class global-input (planned upstream).

**Side-load pre-flight (run on the box before `ob run`, or methods stall/fail):**

| artifact | task | feeds | if missing |
|---|---|---|---|
| `data/godata/gene2go_all.pkl` | `pixi run fetch-godata` | gears/cpa/additive/split/**scfoundation** | GEARS re-downloads (slow) or scfoundation hard-fails |
| `data/godata/go_essential_all.csv` | `pixi run fetch-go-essential` | **scfoundation** | forked GEARS rebuilds the GO graph via a ~99M-pair single-thread loop (**HOURS**) |
| `data/replogle/replogle_k562_essential.h5ad` | `pixi run fetch-replogle` | **transfer**, lpm_k562PertEmb | hard-fail (exit 3); it's the reference dataset the method regresses against |
| `data/replogle/replogle_rpe1_essential.h5ad` | `pixi run fetch-replogle-rpe1` (Dataverse 7458694, md5-pinned 2026-10-05) | lpm_rpe1PertEmb (via `make-pert-emb-rpe1`) | hard-fail (exit 3) |
| `data/embeddings/*.tsv` | `pixi run -e r make-pert-emb-{gears,k562,rpe1}` | **lpm_{gears,k562,rpe1}PertEmb** | hard-fail (exit 3); built from go_essential / Replogle, so fetch those first |
| `data/embeddings/{scgpt,scfoundation}_gene.tsv` | `pixi run -e scgpt-gpu make-gene-emb-scgpt` / `-e scfoundation-gpu make-gene-emb-scfoundation` | **lpm_{scgpt,scFoundation}GeneEmb** | hard-fail (exit 3); CPU weight lookups from the two checkpoints, so fetch those first |
| `data/scfoundation/models.ckpt` | `pixi run fetch-scfoundation-model` (or scp) | **scfoundation** | hard-fail |
| `data/uce/model_files/` | `pixi run -e omnidata fetch-uce-model` (figshare 24320806 via hapiq, md5-pinned; ~15GB, `OMNI_UCE_MODELS=4layers` for one) | **uce**, **uce33** | hard-fail (exit 3) |
| `data/geneformer/geneformer_hf.json` | `pixi run -e hf fetch-geneformer-hf` (HF `ctheodoris/Geneformer@01d3ea89`, code + 95M weights) | **geneformer** | hard-fail (exit 3); a manifest into the shared HF cache — keep that cache on persistent disk |
| `data/scbert/scbert_hf.json` | `pixi run -e hf fetch-scbert-hf` (our GPL-3.0 mirror `btraven/scbert-panglao-pretrain@cbf8b4eb`, sha256-pinned to the Drive copy — a third-party re-upload, see the fetcher header) | **scbert** (not ported yet) | — |
| `data/scgpt/scgpt_human_hf.json` | `pixi run -e hf fetch-scgpt-model-hf` (Hugging Face, via omni-huggingface) | **scgpt** | hard-fail; a manifest into the shared HF cache, so re-run it if the cache is cleared |
| `data/scgpt/scGPT_human/` | `OMNI_SCGPT_URL='file:///abs/scGPT_human' pixi run fetch-scgpt-model` — **deprecated**, same bytes | **scgpt** (fallback) | only needed if the HF repo is unreachable |

`pixi run fetch-all` runs every fetcher above in order (idempotent; `OMNI_FETCH_SKIP="uce …"` to skip some).

**Mirroring between our machines: hapiq's cache.** The Dataverse fetchers (gene2go,
go_essential, Replogle K562/RPE1) download through `modules/godata/hapiq_get.sh`, i.e.
`hapiq download url` with hapiq's content-addressed cache on (`HAPIQ_CACHE_DIR`, default
`~/.cache/hapiq`). A cached URL is served with no network, a cache dir can be copied to
another machine, and `hapiq cache serve` shares it over HTTP: set
`OMNI_HAPIQ_PEERS=http://host:port` (+ `OMNI_HAPIQ_TOKEN`) and peers are asked before the
origin; the client verifies the sha256 while streaming, and the md5 pin is still checked.
- **Canonical cache: `roland:~/.cache/hapiq`**, filled from Dataverse, served by tmux
  `hqserve` on roland:7777 with a token in `roland:~/.config/hapiq-serve.token`. Port 7777
  is firewalled (only ssh gets through), so reach it via a tunnel:
  `ssh -N -L 17777:127.0.0.1:7777 roland` + `OMNI_HAPIQ_PEERS=http://127.0.0.1:17777`.
- **hapiq >= 0.1.0** (`fetch` was removed; `download url` replaces it). The helper uses the
  omnidata env's pinned hapiq before PATH — roland's `/usr/local/bin/hapiq` is an old dev
  build whose `download url` silently writes nothing.
- **pixi version:** install the envs with the pixi that wrote `pixi.lock` (0.71.x). pixi
  0.76 reads this lock but installs the non-CUDA envs (`default`, `hf`, `omnidata`) EMPTY,
  without an error (it moved CUDA from `[system-requirements]` to platform virtual
  packages). On roland use `~/bin/pixi-0.71.2`.

**Mirroring (the hash is the identity, the host is a parameter).** Every fetcher above
takes `OMNI_<NAME>_URL` / `OMNI_<NAME>_MD5` overrides and verifies the hash *after* the
download, so any mirror that serves the canonical bytes passes the same check. Harvard
Dataverse WAF-challenges some networks (HTTP 202, `x-amzn-waf-action: challenge`, empty
body — it hit every datafile URL from the dev box on 2026-08-13), so the three Dataverse
artifacts are mirrored to a Hugging Face **dataset** repo, `btraven/omni-scfm-data`.
Seed it once from a network Dataverse answers:

```bash
hf repos create btraven/omni-scfm-data --repo-type dataset         # once
hf upload btraven/omni-scfm-data <file> --repo-type dataset        # per artifact
# (`pixi run -e hf …`; the hf env has the CLI + an authenticated token)
```

For anyone outside our machines, a public HF repo serves plain HTTPS, so **no new code
or module is needed** — the fetchers take the resolve URL as-is, hash pins unchanged:

```bash
OMNI_GENE2GO_URL=https://huggingface.co/datasets/btraven/omni-scfm-data/resolve/main/gene2go_all.pkl
```

Flip the defaults in the three `run.sh` scripts once the repo is populated, keeping the
Dataverse URL in a comment as the provenance record (the pattern
`modules/godata/scgpt_model_hf` already follows for the scGPT weights, where the sha256
of the canonical Drive release is what makes the mirror trustworthy).

Note the version trap behind go_essential: stock cell-gears 0.1.2 *downloads* it
(Dataverse 6934319) and parallelizes the fallback; scFoundation's forked GEARS 0.0.2
does neither — so its method MUST get the precomputed file.

## Memory

Project-specific gotchas, repro results, and env quirks are kept as agent memory
under the user's memory dir (indexed in `MEMORY.md`), not duplicated here. AGENTS.md
is for stable *conventions*; memory is for *findings*.
