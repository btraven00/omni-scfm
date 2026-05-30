"""omni-scfm results dashboard (stub).

Reproduces the paper's headline single-perturbation view from our own
`scores.parquet`, and overlays our numbers against the paper's published values
to show the reproduction is faithful.

Run locally:  pixi run dashboard
(stlite/GitHub Pages deployment is the M4 follow-up; this app is written
stlite-compatible — pandas + altair, data loaded from files, no server-only deps.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Local dev: make the shared library importable. (For stlite we'd bundle these.)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from omni_scfm.dashboard_data import (  # noqa: E402
    METRIC_LABELS,
    leaderboard,
    load_scores,
    reproduction,
    reproduction_summary,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = REPO / "out" / "scores.parquet"
DEFAULT_PUBLISHED = REPO / "scratch" / "published" / "published_single.csv"

st.set_page_config(page_title="omni-scfm", layout="wide")
st.title("omni-scfm — perturbation prediction benchmark")
st.caption("Reproduction of Ahlmann-Eltze, Huber & Anders (Nat Methods 2025)")

# --- inputs -----------------------------------------------------------------
scores_path = st.sidebar.text_input("scores.parquet", str(DEFAULT_SCORES))
if not Path(scores_path).exists():
    st.warning(f"No scores at `{scores_path}` — run the benchmark then `pixi run collect`.")
    st.stop()

df = load_scores(scores_path)
metric = st.sidebar.selectbox("Metric", list(METRIC_LABELS), format_func=METRIC_LABELS.get)
split = st.sidebar.selectbox("Split", ["test", "val", "train"], index=0)
mlabel = METRIC_LABELS[metric]

st.sidebar.markdown(
    f"**{len(df)}** rows · {df['method'].nunique()} method(s) · "
    f"{df['dataset'].nunique()} dataset(s) · seeds {sorted(df['seed'].dropna().unique())}"
)

# --- leaderboard ------------------------------------------------------------
st.header("Leaderboard")
lb = leaderboard(df, split=split, metric=metric)
st.dataframe(
    lb.style.format({"median": "{:.3f}", "mean": "{:.3f}", "std": "{:.3f}"}),
    use_container_width=True,
)

# --- per-method distribution (the paper's main panel) -----------------------
st.header(f"{mlabel} on held-out ({split}) perturbations")
sub = df[df["split"] == split].dropna(subset=[metric])
order = lb.sort_values("median", ascending=False)["method"].tolist()
base = alt.Chart(sub)
box = base.mark_boxplot(opacity=0.4, color="#bbb").encode(
    x=alt.X("method:N", sort=order, title=None),
    y=alt.Y(f"{metric}:Q", title=mlabel),
)
pts = base.mark_circle(size=35, opacity=0.6).encode(
    x=alt.X("method:N", sort=order),
    y=f"{metric}:Q",
    color=alt.Color("method:N", legend=None),
    tooltip=["method", "perturbation", alt.Tooltip(f"{metric}:Q", format=".3f"), "seed"],
)
st.altair_chart((box + pts).facet(column=alt.Column("dataset:N")).resolve_scale(y="shared"),
                use_container_width=True)

# --- reproduction vs published ---------------------------------------------
st.header("Reproduction vs published")
pub_path = st.sidebar.text_input("published reference (csv)", str(DEFAULT_PUBLISHED))
if not Path(pub_path).exists():
    st.info("No published reference found — run `python scripts/extract_published_numbers.py` "
            "to enable the ours-vs-paper overlay.")
else:
    published = pd.read_csv(pub_path)
    rep = reproduction(df, published, metric=metric)
    if rep.empty:
        st.info("No overlapping (dataset, method, perturbation) between scores and published.")
    else:
        summary = reproduction_summary(rep, metric=metric)
        st.dataframe(
            summary.style.format({"max_abs_diff": "{:.2e}", "mean_abs_diff": "{:.2e}", "corr": "{:.6f}"}),
            use_container_width=True,
        )
        lim = pd.concat([rep[f"{metric}_paper"], rep[f"{metric}_ours"]])
        line = pd.DataFrame({"v": [lim.min(), lim.max()]})
        scatter = alt.Chart(rep).mark_circle(size=40, opacity=0.6).encode(
            x=alt.X(f"{metric}_paper:Q", title=f"{mlabel} (paper)"),
            y=alt.Y(f"{metric}_ours:Q", title=f"{mlabel} (ours)"),
            color="method:N",
            tooltip=["method", "perturbation", "seed",
                     alt.Tooltip("abs_diff:Q", format=".2e")],
        )
        diag = alt.Chart(line).mark_line(color="#999", strokeDash=[4, 4]).encode(x="v:Q", y="v:Q")
        st.altair_chart((diag + scatter).interactive(), use_container_width=True)
