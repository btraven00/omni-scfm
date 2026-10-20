# Run tracking (Weights & Biases)

Optional. With no config file, every method runs exactly as before.

## Turn it on

```bash
cp track.example.yaml track.yaml   # in the repo that holds out/; track.yaml is git-ignored
wandb login                        # once per machine (~/.netrc), or export WANDB_API_KEY
```

`track.yaml` is flat `key: value`:

| key | meaning |
|---|---|
| `project` | W&B project (default `omni-scfm`) |
| `entity` | user or team; empty = the API key's default |
| `mode` | `online`, `offline` (runs saved under `<repo>/wandb/`; upload with `wandb sync`), `disabled` |
| `group` | optional run group, e.g. `gpubox-2026-10` |
| `tags` | optional, comma-separated; `<method>,<dataset>,seed<N>` are always added |

Point at another file with `OMNI_TRACK_CONFIG=/path/track.yaml`.

**Why a file and not the plan:** OmniBenchmark can't feed one global input into every
dataset lineage. 0.5.1 silently drops it, and 0.7.0 rejects the plan. A plan parameter
would re-hash every output path. So `track.yaml` is side-loaded like `data/`.

## How it works

- `modules/methods/track.sh` is sourced by a method's `run.sh` once `$DATA_ROOT`, `$ds`
  and `$seed` are known:
  `source "$REPO/modules/methods/track.sh" <method>`.
  It exports the standard `WANDB_*` variables:
  - run name `<dataset>-seed<N>-<method>`
  - project, entity, mode, group and tags
  - `WANDB_DIR`, which points outside the job's temporary sandbox

  It also sets `OMNI_TRACK=1`. Any `wandb.init()` in the job picks these up.
- `modules/methods/omni_track.py` is what training loops call:
  ```python
  import omni_track
  omni_track.log(step=epoch, train_loss=l, val_mse=m)
  omni_track.finish()
  ```
  It does nothing when tracking is off, or when wandb isn't installed in that env (it prints
  one warning). That makes it safe to sed into vendored code. If the vendored code already
  opened a run, `omni_track` reuses it.
- wandb doesn't touch the python, numpy or torch RNG state (checked on scFoundation), so
  results are the same with tracking on or off.

## Status per method

| method | wandb in env | what gets logged |
|---|---|---|
| scfoundation | 0.30.0 | the fork's own hooks: per-step loss, per-epoch train/val MSE, test metrics |
| scgpt, scbert, uce, cpa, gears (gpu) | 0.30.0 | nothing yet (loops not wired) |
| geneformer | 0.26.1 (the paper pins click 8.1.7; wandb ≥ 0.27 needs click ≥ 8.2) | nothing yet |
| R methods, gears (CPU split/additive) | — | not planned |

To wire a method:
1. Source `track.sh` in its `run.sh`.
2. Add `omni_track.log(...)` calls to its loop (a sed on the vendored script, like the
   other cluster-path patches).
3. Check the predictions are byte-identical with tracking on and off.
