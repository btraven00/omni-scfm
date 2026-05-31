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
    load_published,
    load_scatter,
    load_scores,
    hardest_table,
    method_comparison,
    per_perturbation,
    reproduction,
    reproduction_summary,
    violin_density,
    violin_jitter,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = REPO / "out" / "scores.parquet"
DEFAULT_SCATTER = REPO / "out" / "scatter.parquet"
DEFAULT_PUBLISHED = REPO / "scratch" / "published" / "published_single.csv"
DEFAULT_PUBLISHED_DOUBLE = REPO / "scratch" / "published" / "published_double.csv"

st.set_page_config(page_title="omni-scfm", layout="wide")
st.title("omni-scfm — perturbation prediction benchmark")
st.caption("Reproduction of Ahlmann-Eltze, Huber & Anders (Nat Methods 2025)")

# --- inputs -----------------------------------------------------------------
scores_path = st.sidebar.text_input("scores.parquet", str(DEFAULT_SCORES))
if not Path(scores_path).exists():
    st.warning(f"No scores at `{scores_path}` — run the benchmark then `pixi run collect`.")
    st.stop()

df = load_scores(scores_path)

all_datasets = sorted(df["dataset"].unique())
datasets_sel = st.sidebar.multiselect("Datasets", all_datasets, default=all_datasets)
if not datasets_sel:
    st.warning("Select at least one dataset.")
    st.stop()
df = df[df["dataset"].isin(datasets_sel)]

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

# --- per-perturbation difficulty heatmap + hardest table --------------------
# One tab per dataset: difficulty rankings only make sense within a dataset
# (perturbations and gene panels differ), so we never mix datasets here.
st.header("Per-perturbation difficulty")
pp = per_perturbation(df, split=split, metric=metric)
if pp.empty:
    st.info("No scored perturbations on this split.")
else:
    datasets = sorted(pp["dataset"].unique())
    for tab, ds in zip(st.tabs(datasets), datasets):
        with tab:
            ppd = pp[pp["dataset"] == ds]
            pert_order = ppd.groupby("perturbation")[metric].mean().sort_values().index.tolist()
            n_pert = len(pert_order)
            show_labels = n_pert <= 40            # hide x labels when there are many (Norman)
            # Explicit width per perturbation so every column is visible (a concat
            # chart won't auto-fit to the container, and squeezing hides cells).
            panel_w = max(900, 20 * n_pert)

            # Two selections on `perturbation`, both sourced from the heatmap grid:
            #   hover -> transient highlight (dims other points in the strip above)
            #   pin   -> click to fix a perturbation; persists until cleared.
            # The pin is pure client-side Vega state; the Clear button resets it by
            # bumping a nonce in the chart key, which remounts the Vega view empty.
            nonce_key = f"pin_nonce_{ds}"
            st.session_state.setdefault(nonce_key, 0)
            if st.button("Clear pinned perturbation", key=f"clear_{ds}"):
                st.session_state[nonce_key] += 1

            hover = alt.selection_point(fields=["perturbation"], on="pointerover",
                                        clear="pointerout", empty=True)
            pin = alt.selection_point(fields=["perturbation"], on="click",
                                      toggle=False, clear=False, empty=False)
            xp = alt.X("perturbation:N", sort=pert_order, title=None,
                       axis=alt.Axis(labels=show_labels, labelAngle=-45, ticks=show_labels))

            # top: each perturbation's score, one point per method. Hover dims the
            # rest; the pinned perturbation's points are enlarged so it stays visible.
            strip = alt.Chart(ppd).mark_circle().encode(
                x=xp, xOffset=alt.XOffset("method:N"),
                y=alt.Y(f"{metric}:Q", title=mlabel, scale=alt.Scale(zero=False)),
                color=alt.Color("method:N", scale=alt.Scale(scheme="tableau10")),
                opacity=alt.condition(hover, alt.value(0.95), alt.value(0.12)),
                size=alt.condition(pin, alt.value(220), alt.value(80)),
                tooltip=["perturbation", "method", alt.Tooltip(f"{metric}:Q", format=".3f")],
            ).properties(height=300, width=panel_w, title=f"{mlabel} — {ds}")

            # bottom: difficulty heatmap (hover + click source); pinned column outlined
            heat = alt.Chart(ppd).mark_rect().encode(
                x=alt.X("perturbation:N", sort=pert_order, title="perturbation (hardest → easiest)",
                        axis=alt.Axis(labels=show_labels, labelAngle=-45, ticks=show_labels)),
                y=alt.Y("method:N", title=None),
                color=alt.Color(f"{metric}:Q", title=mlabel,
                                scale=alt.Scale(scheme="redyellowgreen", domainMid=0)),
                stroke=alt.condition(pin, alt.value("#111"), alt.value("#fff")),
                strokeWidth=alt.condition(pin, alt.value(2.5), alt.value(0.4)),
                tooltip=["dataset", "perturbation", "method", alt.Tooltip(f"{metric}:Q", format=".3f")],
            ).add_params(hover, pin).properties(height=64 * ppd["method"].nunique() + 30, width=panel_w)

            st.altair_chart(
                alt.vconcat(strip, heat).resolve_scale(x="shared", color="independent"),
                use_container_width=False,
                key=f"diff_{ds}_{metric}_{split}_{st.session_state[nonce_key]}",
            )

            st.subheader("Hardest perturbations")
            ht = hardest_table(ppd, metric=metric).drop(columns="dataset")
            numcols = [c for c in ht.columns if c != "perturbation"]
            st.dataframe(ht.style.format({c: "{:.3f}" for c in numcols}),
                         use_container_width=True, height=360)

# --- paper figure: L2 beeswarm + linked predicted-vs-observed scatter grid --
# Reproduces the paper's double-perturbation figure: (a) a beeswarm of L2 error
# per method, (b) per-method predicted-Δ vs observed-Δ scatter for one example
# perturbation. Pick the example in the selectbox -> it's highlighted (orange) in
# the beeswarm and drives the scatter grid.
st.header("Double-perturbation prediction error")
scatter_path = st.sidebar.text_input("scatter.parquet", str(DEFAULT_SCATTER))
scatter_df = load_scatter(scatter_path) if Path(scatter_path).exists() else None
ppl2 = per_perturbation(df, split="test", metric="l2")

for tab, ds in zip(st.tabs(datasets_sel), datasets_sel):
    with tab:
        pp_ds = ppl2[ppl2["dataset"] == ds].copy()
        if pp_ds.empty:
            st.info("No scored test perturbations for this dataset.")
            continue
        perts = pp_ds.groupby("perturbation")["l2"].mean().sort_values(
            ascending=False).index.tolist()  # hardest first
        chosen = st.selectbox("Example perturbation (highlight + scatter)", perts,
                              key=f"ex_{ds}")
        pp_ds["sel"] = pp_ds["perturbation"] == chosen
        morder = pp_ds.groupby("method")["l2"].mean().sort_values().index.tolist()
        centers = {m: i for i, m in enumerate(morder)}

        # (a) violin (KDE) per method with the raw perturbation points jittered on top.
        # The jitter is random *inside the violin envelope* (wide where dense), which
        # avoids the columnar alignment a swarm produces. Red segment = mean, orange =
        # the chosen example. Everything is on an explicit numeric x (method centres),
        # since Altair's xOffset:Q doesn't expand to the band width.
        HW = 0.42
        pp_ds["xpos"] = 0.0
        viol = []
        for m in morder:
            mask = pp_ds["method"] == m
            vals = pp_ds.loc[mask, "l2"].to_numpy()
            g = violin_density(vals)
            g["method"] = m
            g["xL"] = centers[m] - g["dens"] * HW
            g["xR"] = centers[m] + g["dens"] * HW
            viol.append(g)
            pp_ds.loc[mask, "xpos"] = centers[m] + violin_jitter(vals, halfwidth=HW,
                                                                 seed=centers[m])
        viol = pd.concat(viol, ignore_index=True)
        xscale = alt.Scale(domain=[-0.5, len(morder) - 0.5])
        label_expr = "[" + ",".join(f"'{m}'" for m in morder) + "][datum.value]"
        xaxis = alt.Axis(values=list(range(len(morder))), labelExpr=label_expr,
                         labelAngle=-30, labelFontSize=12, grid=False, tickSize=0)

        violin = alt.Chart(viol).mark_area(opacity=0.25, color="#6c8ebf").encode(
            x=alt.X("xL:Q", scale=xscale, title=None, axis=None), x2="xR:Q",
            y=alt.Y("y:Q", title="Prediction error (L2)"), detail="method:N")
        pts = alt.Chart(pp_ds).mark_circle().encode(
            x=alt.X("xpos:Q", title=None, scale=xscale, axis=xaxis),
            y="l2:Q",
            color=alt.Color("sel:N", scale=alt.Scale(domain=[False, True],
                            range=["#9aa0a6", "orange"]), legend=None),
            size=alt.Size("sel:N", scale=alt.Scale(domain=[False, True],
                          range=[26, 170]), legend=None),
            order=alt.Order("sel:N"),   # draw the orange point on top
            tooltip=["method", "perturbation", alt.Tooltip("l2:Q", format=".2f")],
        )
        means = pp_ds.groupby("method", as_index=False)["l2"].mean()
        mean_seg = pd.concat([
            pd.DataFrame({"method": m, "x": [centers[m] - 0.3, centers[m] + 0.3], "l2": [v, v]})
            for m, v in zip(means["method"], means["l2"])
        ], ignore_index=True)
        mean_layer = alt.Chart(mean_seg).mark_line(color="red", size=2).encode(
            x=alt.X("x:Q", scale=xscale, axis=None), y="l2:Q", detail="method:N")
        st.altair_chart((violin + pts + mean_layer).properties(height=420),
                        use_container_width=True)

        # (b) predicted-Δ vs observed-Δ scatter, one facet per method, for `chosen`
        if scatter_df is None:
            st.info("No scatter.parquet — run `pixi run collect` to enable the scatter grid.")
            continue
        sc = scatter_df[(scatter_df["dataset"] == ds)
                        & (scatter_df["perturbation"] == chosen)
                        & (scatter_df["method"].isin(morder))]  # same methods as beeswarm
        if sc.empty:
            st.info(f"No per-gene scatter points for {chosen}.")
            continue
        lim = float(max(sc["observed_delta"].abs().max(), sc["predicted_delta"].abs().max()))
        # points + the y=x diagonal in ONE dataframe (facet of layered charts needs a
        # shared data source); a `layer` flag selects each mark, `method` keeps facets.
        diag = pd.concat([
            pd.DataFrame({"method": m, "observed_delta": [-lim, lim],
                          "predicted_delta": [-lim, lim], "gene": None})
            for m in sc["method"].unique()
        ], ignore_index=True)
        combined = pd.concat([sc.assign(layer="pt"), diag.assign(layer="diag")],
                             ignore_index=True)
        enc_x = alt.X("observed_delta:Q", title="observed − control",
                      scale=alt.Scale(domain=[-lim, lim]))
        enc_y = alt.Y("predicted_delta:Q", title="predicted − control",
                      scale=alt.Scale(domain=[-lim, lim]))
        base = alt.Chart(combined)
        line = base.transform_filter("datum.layer == 'diag'").mark_line(
            color="#bbb", strokeDash=[4, 4]).encode(x=enc_x, y=enc_y, detail="method:N")
        scat = base.transform_filter("datum.layer == 'pt'").mark_circle(
            size=12, opacity=0.35).encode(
            x=enc_x, y=enc_y,
            tooltip=["gene", alt.Tooltip("observed_delta:Q", format=".2f"),
                     alt.Tooltip("predicted_delta:Q", format=".2f")])
        grid = (line + scat).properties(width=180, height=180).facet(
            facet=alt.Facet("method:N", sort=morder, title=None), columns=4)
        st.altair_chart(grid, use_container_width=False)

        # R²δ / L2 annotation per method for the chosen perturbation (from scores).
        # The paper's panel-b "R²" is actually the Pearson corr of predicted-Δ vs
        # observed-Δ (i.e. our pearson_delta), NOT its square — show it directly so
        # the numbers match the paper.
        ann = (df[(df["dataset"] == ds) & (df["perturbation"] == chosen)
                  & (df["split"] == "test")]
               .groupby("method")[["pearson_delta", "l2"]].mean())
        if not ann.empty:
            ann = ann.rename(columns={"pearson_delta": "R²δ", "l2": "L2"}) \
                     .reindex(morder).dropna(how="all")
            st.caption(f"Example: {chosen}  (R²δ = Pearson corr of predicted-Δ vs observed-Δ)")
            st.dataframe(ann.style.format({"R²δ": "{:.3f}", "L2": "{:.2f}"}),
                         use_container_width=False)

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
# Compare our numbers to the paper's, for single- AND double-perturbation
# datasets at once: the single table covers adamson, the double table covers
# norman_from_scfoundation (method names normalised to our ids in load_published).
st.header("Reproduction vs published")
pub_single = st.sidebar.text_input("published single (csv)", str(DEFAULT_PUBLISHED))
pub_double = st.sidebar.text_input("published double (csv)", str(DEFAULT_PUBLISHED_DOUBLE))
pub_paths = [p for p in (pub_single, pub_double) if Path(p).exists()]
if not pub_paths:
    st.info("No published reference found — run `python scripts/extract_published_numbers.py` "
            "to enable the ours-vs-paper overlay.")
else:
    published = load_published(pub_paths)
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
