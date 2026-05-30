# Dashboard

Streamlit results explorer for `scores.parquet`: a leaderboard, the paper's
per-method Pearson-delta panel, and an ours-vs-published reproduction overlay.

## Run

```bash
ob run benchmark.yaml --dirty   # produce results
pixi run collect                # -> out/scores.parquet
python scripts/extract_published_numbers.py   # (optional) enables the repro overlay
pixi run dashboard              # streamlit on out/scores.parquet
```

Point it at any run via the sidebar (`scores.parquet` path).

## Notes

- Written **stlite-compatible** (pandas + altair, data loaded from files, no
  server-only deps) so it can later be published to GitHub Pages via stlite
  (M4). For now it runs as a normal local Streamlit app.
- Data helpers live in `omni_scfm.dashboard_data` (unit-tested); `app.py` is the
  thin UI layer.
