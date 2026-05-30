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
    method_comparison,
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
sub = df[df["split"] == split].dropna(subset=[metric]).copy()
order = lb.sort_values("median", ascending=False)["method"].tolist()
n_methods, n_datasets = sub["method"].nunique(), sub["dataset"].nunique()

x = alt.X("method:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30, labelFontSize=12))
y = alt.Y(f"{metric}:Q", title=mlabel, scale=alt.Scale(zero=False))
base = alt.Chart(sub)
box = base.mark_boxplot(size=40, opacity=0.35, color="#aaa", outliers=False).encode(x=x, y=y)
pts = base.mark_circle(size=70, opacity=0.55).encode(
    x=x,
    xOffset="jitter:Q",                       # spread points so they don't stack
    y=y,
    color=alt.Color("method:N", legend=None, scale=alt.Scale(scheme="tableau10")),
    tooltip=["method", "perturbation", alt.Tooltip(f"{metric}:Q", format=".3f"), "seed"],
).transform_calculate(jitter="(random() - 0.5) * 0.6")
# dashed line at the best method's mean (paper aesthetic)
rule = base.mark_rule(color="#444", strokeDash=[4, 4]).encode(
    y=f"mean({metric}):Q"
).transform_filter(alt.FieldEqualPredicate(field="method", equal=order[0]))

chart = (box + pts + rule).properties(height=460)
if n_datasets > 1:
    chart = chart.properties(width=max(220, 90 * n_methods)).facet(column="dataset:N")
    st.altair_chart(chart.resolve_scale(y="shared"), use_container_width=False)
else:
    st.altair_chart(chart, use_container_width=True)   # stretch full width

# --- method-vs-method per-perturbation comparison ---------------------------
if n_methods >= 2:
    st.header("Method comparison (per perturbation)")
    methods = sorted(sub["method"].unique())
    c1, c2 = st.columns(2)
    mx = c1.selectbox("x-axis method", methods, index=0)
    my = c2.selectbox("y-axis method", methods, index=1)
    comp = method_comparison(df, mx, my, metric=metric, split=split)
    if comp.empty:
        st.info("No shared perturbations between the two methods on this split.")
    else:
        wins = comp["winner"].value_counts().to_dict()
        st.caption(f"{my} better on {wins.get(my, 0)} / {len(comp)} perturbations "
                   f"(diagonal = tie; above = {my} wins).")
        lim = pd.concat([comp["x"], comp["y"]])
        dline = pd.DataFrame({"v": [lim.min(), lim.max()]})
        sc = alt.Chart(comp).mark_circle(size=70, opacity=0.6).encode(
            x=alt.X("x:Q", title=f"{mlabel} — {mx}", scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", title=f"{mlabel} — {my}", scale=alt.Scale(zero=False)),
            color=alt.Color("winner:N", title="better"),
            tooltip=["perturbation", "seed",
                     alt.Tooltip("x:Q", format=".3f"), alt.Tooltip("y:Q", format=".3f")],
        )
        diag = alt.Chart(dline).mark_line(color="#999", strokeDash=[4, 4]).encode(x="v:Q", y="v:Q")
        st.altair_chart((diag + sc).properties(height=460).interactive(), use_container_width=True)

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
