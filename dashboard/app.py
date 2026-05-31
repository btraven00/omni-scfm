"""omni-scfm results dashboard.

Two top-level tabs keep a clear line between reproduction and our own work:
  • 📄 Paper reproduction — faithful views of the paper (leaderboard, the held-out
    per-method distribution, the double-perturbation figure 1a/b, and the
    ours-vs-published overlay).
  • 🧪 Extensions — analyses beyond the paper (per-perturbation difficulty,
    method comparison, and OmniPath gene-network centrality vs difficulty).

Run locally:  pixi run dashboard
(stlite-compatible: pandas + altair, data loaded from files, no server-only deps.)
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
    CENTRALITY_MEASURES,
    METRIC_LABELS,
    hardest_table,
    leaderboard,
    load_centrality,
    load_published,
    load_scatter,
    load_scores,
    method_comparison,
    per_perturbation,
    perturbation_centrality,
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
DEFAULT_CENTRALITY = REPO / "data" / "gene_centrality.csv"

st.set_page_config(page_title="omni-scfm", layout="wide")
st.title("omni-scfm — perturbation prediction benchmark")
st.caption("Reproduction of Ahlmann-Eltze, Huber & Anders (Nat Methods 2025) + extensions")

# --- sidebar inputs (shared by both tabs) -----------------------------------
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
scatter_path = st.sidebar.text_input("scatter.parquet", str(DEFAULT_SCATTER))
pub_single = st.sidebar.text_input("published single (csv)", str(DEFAULT_PUBLISHED))
pub_double = st.sidebar.text_input("published double (csv)", str(DEFAULT_PUBLISHED_DOUBLE))
centrality_path = st.sidebar.text_input("gene_centrality.csv", str(DEFAULT_CENTRALITY))

st.sidebar.markdown(
    f"**{len(df)}** rows · {df['method'].nunique()} method(s) · "
    f"{df['dataset'].nunique()} dataset(s) · seeds {sorted(df['seed'].dropna().unique())}"
)

# shared artifacts + derived frames (used across tabs)
scatter_df = load_scatter(scatter_path) if Path(scatter_path).exists() else None
centrality = load_centrality(centrality_path) if Path(centrality_path).exists() else None
sub = df[df["split"] == split].dropna(subset=[metric]).copy()
lb = leaderboard(df, split=split, metric=metric)
order = lb.sort_values("median", ascending=False)["method"].tolist()
n_methods, n_datasets = sub["method"].nunique(), sub["dataset"].nunique()

tab_repro, tab_ext = st.tabs(["📄 Paper reproduction", "🧪 Extensions"])

# ============================================================================
# Paper reproduction
# ============================================================================
with tab_repro:
    # --- leaderboard --------------------------------------------------------
    st.header("Leaderboard")
    st.dataframe(
        lb.style.format({"median": "{:.3f}", "mean": "{:.3f}", "std": "{:.3f}"}),
        use_container_width=True,
    )

    # --- per-method distribution (the paper's main panel) -------------------
    st.header(f"{mlabel} on held-out ({split}) perturbations")
    if sub.empty:
        st.info(f"No scored perturbations on the {split} split.")
    else:
        x = alt.X("method:N", sort=order, title=None,
                  axis=alt.Axis(labelAngle=-30, labelFontSize=12))
        y = alt.Y(f"{metric}:Q", title=mlabel, scale=alt.Scale(zero=False))
        base = alt.Chart(sub)
        box = base.mark_boxplot(size=40, opacity=0.35, color="#aaa", outliers=False).encode(x=x, y=y)
        pts = base.mark_circle(size=70, opacity=0.55).encode(
            x=x, xOffset="jitter:Q", y=y,
            color=alt.Color("method:N", legend=None, scale=alt.Scale(scheme="tableau10")),
            tooltip=["method", "perturbation", alt.Tooltip(f"{metric}:Q", format=".3f"), "seed"],
        ).transform_calculate(jitter="(random() - 0.5) * 0.6")
        rule = base.mark_rule(color="#444", strokeDash=[4, 4]).encode(
            y=f"mean({metric}):Q"
        ).transform_filter(alt.FieldEqualPredicate(field="method", equal=order[0]))
        chart = (box + pts + rule).properties(height=460)
        if n_datasets > 1:
            chart = chart.properties(width=max(220, 90 * n_methods)).facet(column="dataset:N")
            st.altair_chart(chart.resolve_scale(y="shared"), use_container_width=False)
        else:
            st.altair_chart(chart, use_container_width=True)

    # --- double-perturbation figure: L2 violin + linked scatter grid --------
    # Reproduces the paper's double-pert figure: (a) violin/beeswarm of L2 error
    # per method, (b) per-method predicted-Δ vs observed-Δ scatter for one example
    # perturbation. Click a point (or use the dropdown) to drive the scatter.
    st.header("Double-perturbation prediction error")
    ppl2 = per_perturbation(df, split="test", metric="l2")
    for tab, ds in zip(st.tabs(datasets_sel), datasets_sel):
        with tab:
            pp_ds = ppl2[ppl2["dataset"] == ds].copy()
            if pp_ds.empty:
                st.info("No scored test perturbations for this dataset.")
                continue
            perts = pp_ds.groupby("perturbation")["l2"].mean().sort_values(
                ascending=False).index.tolist()  # hardest first
            # A click on the beeswarm lands here as a pending value; apply it to the
            # selectbox's state *before* the widget is created (Streamlit forbids
            # mutating a widget's state after instantiation), so dropdown + scatter follow.
            pending = st.session_state.pop(f"pending_{ds}", None)
            if pending in perts:
                st.session_state[f"ex_{ds}"] = pending
            chosen = st.selectbox("Example perturbation (click a point below, or pick)",
                                  perts, key=f"ex_{ds}")
            pp_ds["sel"] = pp_ds["perturbation"] == chosen
            morder = pp_ds.groupby("method")["l2"].mean().sort_values().index.tolist()
            centers = {m: i for i, m in enumerate(morder)}

            # (a) violin (KDE) per method with raw points jittered inside the envelope
            # (no columnar swarm). Red segment = mean, orange = chosen example. Explicit
            # numeric x (method centres), since Altair's xOffset:Q doesn't fill the band.
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
            click_sel = alt.selection_point(name="clickpert", fields=["perturbation"],
                                            on="click", empty=False)
            pts = alt.Chart(pp_ds).mark_circle().encode(
                x=alt.X("xpos:Q", title=None, scale=xscale, axis=xaxis),
                y="l2:Q",
                color=alt.Color("sel:N", scale=alt.Scale(domain=[False, True],
                                range=["#9aa0a6", "orange"]), legend=None),
                size=alt.Size("sel:N", scale=alt.Scale(domain=[False, True],
                              range=[26, 170]), legend=None),
                order=alt.Order("sel:N"),
                tooltip=["method", "perturbation", alt.Tooltip("l2:Q", format=".2f")],
            ).add_params(click_sel)
            means = pp_ds.groupby("method", as_index=False)["l2"].mean()
            mean_seg = pd.concat([
                pd.DataFrame({"method": m, "x": [centers[m] - 0.3, centers[m] + 0.3], "l2": [v, v]})
                for m, v in zip(means["method"], means["l2"])
            ], ignore_index=True)
            mean_layer = alt.Chart(mean_seg).mark_line(color="red", size=2).encode(
                x=alt.X("x:Q", scale=xscale, axis=None), y="l2:Q", detail="method:N")
            event = st.altair_chart((violin + pts + mean_layer).properties(height=420),
                                    use_container_width=True, on_select="rerun",
                                    key=f"bee_{ds}_{metric}")
            picked = None
            esel = getattr(event, "selection", None) or {}
            crows = esel.get("clickpert") if isinstance(esel, dict) else None
            if crows:
                last = crows[-1]
                picked = last.get("perturbation") if isinstance(last, dict) else last
            if picked and picked in perts and picked != chosen:
                st.session_state[f"pending_{ds}"] = picked
                st.rerun()

            # (b) predicted-Δ vs observed-Δ scatter, one facet per method, for `chosen`
            if scatter_df is None:
                st.info("No scatter.parquet — run `pixi run collect` to enable the scatter grid.")
                continue
            sc = scatter_df[(scatter_df["dataset"] == ds)
                            & (scatter_df["perturbation"] == chosen)
                            & (scatter_df["method"].isin(morder))]
            if sc.empty:
                st.info(f"No per-gene scatter points for {chosen}.")
                continue
            lim = float(max(sc["observed_delta"].abs().max(), sc["predicted_delta"].abs().max()))
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
            cbase = alt.Chart(combined)
            line = cbase.transform_filter("datum.layer == 'diag'").mark_line(
                color="#bbb", strokeDash=[4, 4]).encode(x=enc_x, y=enc_y, detail="method:N")
            scat = cbase.transform_filter("datum.layer == 'pt'").mark_circle(
                size=12, opacity=0.35).encode(
                x=enc_x, y=enc_y,
                tooltip=["gene", alt.Tooltip("observed_delta:Q", format=".2f"),
                         alt.Tooltip("predicted_delta:Q", format=".2f")])
            grid = (line + scat).properties(width=180, height=180).facet(
                facet=alt.Facet("method:N", sort=morder, title=None), columns=4)
            st.altair_chart(grid, use_container_width=False)

            # R²δ / L2 annotation per method (R²δ = Pearson corr of predicted-Δ vs
            # observed-Δ, as in the paper's panel b — shown directly, not squared).
            ann = (df[(df["dataset"] == ds) & (df["perturbation"] == chosen)
                      & (df["split"] == "test")]
                   .groupby("method")[["pearson_delta", "l2"]].mean())
            if not ann.empty:
                ann = ann.rename(columns={"pearson_delta": "R²δ", "l2": "L2"}) \
                         .reindex(morder).dropna(how="all")
                st.caption(f"Example: {chosen}  (R²δ = Pearson corr of predicted-Δ vs observed-Δ)")
                st.dataframe(ann.style.format({"R²δ": "{:.3f}", "L2": "{:.2f}"}),
                             use_container_width=False)

    # --- reproduction vs published ------------------------------------------
    # Single table covers adamson, double covers norman_from_scfoundation (method
    # names normalised to our ids in load_published).
    st.header("Reproduction vs published")
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
                summary.style.format({"max_abs_diff": "{:.2e}", "mean_abs_diff": "{:.2e}",
                                      "corr": "{:.6f}"}),
                use_container_width=True,
            )
            lim = pd.concat([rep[f"{metric}_paper"], rep[f"{metric}_ours"]])
            line = pd.DataFrame({"v": [lim.min(), lim.max()]})
            scatter = alt.Chart(rep).mark_circle(size=40, opacity=0.6).encode(
                x=alt.X(f"{metric}_paper:Q", title=f"{mlabel} (paper)"),
                y=alt.Y(f"{metric}_ours:Q", title=f"{mlabel} (ours)"),
                color="method:N",
                tooltip=["method", "perturbation", "seed", alt.Tooltip("abs_diff:Q", format=".2e")],
            )
            diag = alt.Chart(line).mark_line(color="#999", strokeDash=[4, 4]).encode(x="v:Q", y="v:Q")
            st.altair_chart((diag + scatter).interactive(), use_container_width=True)

# ============================================================================
# Extensions (beyond the paper)
# ============================================================================
with tab_ext:
    # --- per-perturbation difficulty (+ OmniPath centrality in the table) ---
    # One tab per dataset: difficulty rankings only make sense within a dataset.
    st.header("Per-perturbation difficulty")
    pp = per_perturbation(df, split=split, metric=metric)
    if pp.empty:
        st.info("No scored perturbations on this split.")
    else:
        for tab, ds in zip(st.tabs(sorted(pp["dataset"].unique())), sorted(pp["dataset"].unique())):
            with tab:
                ppd = pp[pp["dataset"] == ds]
                pert_order = ppd.groupby("perturbation")[metric].mean().sort_values().index.tolist()
                n_pert = len(pert_order)
                show_labels = n_pert <= 40
                panel_w = max(900, 20 * n_pert)

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
                strip = alt.Chart(ppd).mark_circle().encode(
                    x=xp, xOffset=alt.XOffset("method:N"),
                    y=alt.Y(f"{metric}:Q", title=mlabel, scale=alt.Scale(zero=False)),
                    color=alt.Color("method:N", scale=alt.Scale(scheme="tableau10")),
                    opacity=alt.condition(hover, alt.value(0.95), alt.value(0.12)),
                    size=alt.condition(pin, alt.value(220), alt.value(80)),
                    tooltip=["perturbation", "method", alt.Tooltip(f"{metric}:Q", format=".3f")],
                ).properties(height=300, width=panel_w, title=f"{mlabel} — {ds}")
                heat = alt.Chart(ppd).mark_rect().encode(
                    x=alt.X("perturbation:N", sort=pert_order,
                            title="perturbation (hardest → easiest)",
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
                fmt = {c: "{:.3f}" for c in ht.columns if c != "perturbation"}
                if centrality is not None:
                    # enrich with OmniPath centrality of the perturbed genes (max = "has a hub")
                    od = perturbation_centrality(ht["perturbation"], centrality, "out_degree", "max")
                    bt = perturbation_centrality(ht["perturbation"], centrality, "betweenness", "max")
                    ht["out-deg(max)"] = ht["perturbation"].map(od)
                    ht["betw(max)"] = ht["perturbation"].map(bt)
                    fmt.update({"out-deg(max)": "{:.0f}", "betw(max)": "{:.4f}"})
                st.dataframe(ht.style.format(fmt, na_rep="—"),
                             use_container_width=True, height=360)

    # --- method-vs-method per-perturbation comparison -----------------------
    if n_methods >= 2:
        st.header("Method comparison (per perturbation)")
        methods = sorted(sub["method"].unique())
        c1, c2 = st.columns(2)
        mx = c1.selectbox("x-axis method", methods, index=0)
        my = c2.selectbox("y-axis method", methods, index=min(1, len(methods) - 1))
        comp = method_comparison(df, mx, my, metric=metric, split=split)
        if comp.empty:
            st.info("No shared perturbations between the two methods on this split.")
        else:
            wins = comp["winner"].value_counts().to_dict()
            st.caption(f"{my} better on {wins.get(my, 0)} / {len(comp)} perturbations "
                       f"(diagonal = tie; above = {my} wins).")
            lim = pd.concat([comp["x"], comp["y"]])
            dline = pd.DataFrame({"v": [lim.min(), lim.max()]})
            scc = alt.Chart(comp).mark_circle(size=70, opacity=0.6).encode(
                x=alt.X("x:Q", title=f"{mlabel} — {mx}", scale=alt.Scale(zero=False)),
                y=alt.Y("y:Q", title=f"{mlabel} — {my}", scale=alt.Scale(zero=False)),
                color=alt.Color("winner:N", title="better"),
                tooltip=["perturbation", "seed",
                         alt.Tooltip("x:Q", format=".3f"), alt.Tooltip("y:Q", format=".3f")],
            )
            diag = alt.Chart(dline).mark_line(color="#999", strokeDash=[4, 4]).encode(x="v:Q", y="v:Q")
            st.altair_chart((diag + scc).properties(height=460).interactive(),
                            use_container_width=True)

    # --- OmniPath gene-network centrality vs difficulty ---------------------
    # Hypothesis: perturbing high-out-degree (broad regulatory) genes is harder.
    # Difficulty = R²δ (scale-invariant); we also show L2 (magnitude) and the
    # partial correlation controlling for it, since centrality correlates with
    # effect magnitude (the confound we found earlier).
    st.header("Gene-network centrality vs difficulty (OmniPath)")
    if centrality is None:
        st.info(f"No `{Path(centrality_path).name}` — run `pixi run -e omnipath "
                "python scripts/gene_centrality.py` to enable this.")
    else:
        c1, c2 = st.columns(2)
        cds = c1.selectbox("dataset", datasets_sel, key="cent_ds")
        measure = c2.selectbox("centrality measure", CENTRALITY_MEASURES, key="cent_measure")
        d = df[(df["dataset"] == cds) & (df["split"] == split)].dropna(subset=["pearson_delta"])
        g = (d.groupby("perturbation")
             .agg(R2d=("pearson_delta", "mean"), L2=("l2", "mean")).reset_index())
        g["cent"] = g["perturbation"].map(
            perturbation_centrality(g["perturbation"], centrality, measure, "max"))
        g = g.dropna(subset=["cent", "R2d"])
        if len(g) < 3:
            st.info("Too few perturbations with centrality + difficulty to correlate.")
        else:
            rk = g[["cent", "R2d", "L2"]].rank()
            cm = rk.corr()
            r_cd, r_cl, r_dl = cm.loc["cent", "R2d"], cm.loc["cent", "L2"], cm.loc["R2d", "L2"]
            denom = ((1 - r_cl ** 2) * (1 - r_dl ** 2)) ** 0.5
            partial = (r_cd - r_cl * r_dl) / denom if denom > 0 else float("nan")
            st.caption(
                f"Spearman ρ(centrality, R²δ) = **{r_cd:+.2f}**  ·  ρ(centrality, L2 = magnitude) "
                f"= **{r_cl:+.2f}**  ·  partial ρ(centrality, R²δ | L2) = **{partial:+.2f}**  "
                f"(n={len(g)}). R²δ higher = easier, so negative ρ ⇒ central genes harder; "
                f"if the partial ≈ 0 while ρ-with-L2 is strong, it's a magnitude confound.")
            pt = alt.Chart(g).mark_circle(size=80, opacity=0.6).encode(
                x=alt.X("cent:Q", title=f"{measure} (max over the pair's genes)"),
                y=alt.Y("R2d:Q", title="R²δ (mean over methods; higher = easier)"),
                size=alt.Size("L2:Q", title="L2 (magnitude)"),
                tooltip=["perturbation", alt.Tooltip("cent:Q", format=".2f"),
                         alt.Tooltip("R2d:Q", format=".3f"), alt.Tooltip("L2:Q", format=".2f")],
            )
            st.altair_chart(pt.properties(height=420).interactive(), use_container_width=True)
            cov = perturbation_centrality(g["perturbation"], centrality, measure, "max").notna().mean()
            st.caption(f"OmniPath coverage of {cds} perturbations: {cov*100:.0f}%.")
