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
following the rules above.

## Reference data (side-loading)

Global, dataset-independent reference files (e.g. GEARS' `gene2go_all.pkl`) are
**side-loaded**, not OB stages: OB 0.5.1 can't wire one shared artifact into multiple
per-dataset lineages (a parallel-root input is silently dropped from argv). Pattern:
an md5-pinned fetcher under `modules/<group>/<name>/run.sh` + a `pixi run fetch-*`
task that lands it in a git-ignored `data/<group>/` dir; consuming `run.sh` scripts
**default to that conventional path** (with a flag override). Run the fetch once
before `ob run`. Don't try to express it as an OB `inputs:` node until OB grows a
first-class global-input (planned upstream).

## Memory

Project-specific gotchas, repro results, and env quirks are kept as agent memory
under the user's memory dir (indexed in `MEMORY.md`), not duplicated here. AGENTS.md
is for stable *conventions*; memory is for *findings*.
