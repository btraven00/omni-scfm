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

import os
import sys
import tempfile
import urllib.request
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

# Results are consumed from the decoupled results repo (omni-scfm-results) by URL when
# deployed (e.g. Streamlit Cloud), or from local files during dev. Override the base
# with OMNI_RESULTS_BASE. See https://github.com/btraven00/omni-scfm-results .
RESULTS_BASE = os.environ.get(
    "OMNI_RESULTS_BASE",
    "https://raw.githubusercontent.com/btraven00/omni-scfm-results/main",
)


def _src(local: Path, name: str) -> str:
    """Local file if present (dev), else the published results-repo URL (deploy)."""
    return str(local) if local.exists() else f"{RESULTS_BASE}/{name}"


@st.cache_data(show_spinner=False)
def _localize(src: str | None) -> str | None:
    """The loaders take file paths; for an http(s) source, fetch once to a temp file
    (stdlib only — keeps the deploy free of an fsspec/aiohttp dependency)."""
    if not src:
        return None
    if not src.startswith(("http://", "https://")):
        return src if Path(src).exists() else None
    try:
        req = urllib.request.Request(src, headers={"User-Agent": "omni-scfm-dashboard"})
        fd, tmp = tempfile.mkstemp(suffix=Path(src).suffix or ".dat")
        with urllib.request.urlopen(req) as r, os.fdopen(fd, "wb") as f:
            f.write(r.read())
        return tmp
    except Exception:
        return None


DEFAULT_SCORES = _src(REPO / "out" / "scores.parquet", "scores.parquet")
DEFAULT_SCATTER = _src(REPO / "out" / "scatter.parquet", "scatter.parquet")
DEFAULT_PUBLISHED = _src(REPO / "scratch" / "published" / "published_single.csv",
                         "reference/published_single.csv")
DEFAULT_PUBLISHED_DOUBLE = _src(REPO / "scratch" / "published" / "published_double.csv",
                                "reference/published_double.csv")
DEFAULT_CENTRALITY = _src(REPO / "data" / "gene_centrality.csv", "reference/gene_centrality.csv")

st.set_page_config(page_title="omni-scfm", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
    .omni-title { font-family: 'Fraunces', Georgia, serif; font-size: 3.4rem;
                  font-weight: 600; letter-spacing: -0.015em; line-height: 1.05;
                  margin: 0.2rem 0 0.15rem; }
    .omni-sub   { font-family: 'Fraunces', Georgia, serif; font-size: 1.35rem;
                  font-weight: 400; color: #6b7280; margin: 0 0 1.1rem; }
    .omni-sub a { color: #4b5fa6; text-decoration: none; border-bottom: 1px solid #c7cee6; }
    .omni-sub a:hover { color: #2f3f80; }
    </style>
    <div class="omni-title">omni-scfm</div>
    <div class="omni-sub">Reproduction of Ahlmann-Eltze, Huber &amp; Anders
        (<a href="https://doi.org/10.1038/s41592-025-02772-6" target="_blank">Nat&nbsp;Methods&nbsp;2025</a>)
        + extensions</div>
    """,
    unsafe_allow_html=True,
)


# --- sidebar inputs (shared by both tabs) -----------------------------------
scores_path = st.sidebar.text_input("scores.parquet", str(DEFAULT_SCORES))
scores_local = _localize(scores_path)
if scores_local is None:
    st.warning(f"No scores at `{scores_path}` — run the benchmark then `pixi run collect`.")
    st.stop()

df = load_scores(scores_local)
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

# credit — plain sidebar content (lives in the left bar, below the controls)
st.sidebar.divider()
st.sidebar.caption("powered by [OmniBenchmark](https://omnibenchmark.org)")

# shared artifacts + derived frames (used across tabs)
scatter_local = _localize(scatter_path)
scatter_df = load_scatter(scatter_local) if scatter_local else None
centrality_local = _localize(centrality_path)
centrality = load_centrality(centrality_local) if centrality_local else None
sub = df[df["split"] == split].dropna(subset=[metric]).copy()
lb = leaderboard(df, split=split, metric=metric)
order = lb.sort_values("median", ascending=False)["method"].tolist()
n_methods, n_datasets = sub["method"].nunique(), sub["dataset"].nunique()

tab_repro, tab_ext = st.tabs(["📄 Paper reproduction", "🧪 Extensions"])

# ============================================================================
# Paper reproduction
# ============================================================================
with tab_repro:
    # (leaderboard table removed; `lb`/`order` still drive the chart sort below)
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
    st.header(f"Double-perturbation prediction — {mlabel}")
    harder_high = metric == "l2"   # L2: higher = harder; pearson*: higher = easier
    ppm = per_perturbation(df, split=split, metric=metric)
    for tab, ds in zip(st.tabs(datasets_sel), datasets_sel):
        with tab:
            pp_ds = ppm[ppm["dataset"] == ds].copy()
            if pp_ds.empty:
                st.info(f"No scored {split} perturbations for this dataset.")
                continue
            perts = pp_ds.groupby("perturbation")[metric].mean().sort_values(
                ascending=not harder_high).index.tolist()  # hardest first
            # A click on the beeswarm lands here as a pending value; apply it to the
            # selectbox's state *before* the widget is created (Streamlit forbids
            # mutating a widget's state after instantiation), so dropdown + scatter follow.
            pending = st.session_state.pop(f"pending_{ds}", None)
            if pending in perts:
                st.session_state[f"ex_{ds}"] = pending
            chosen = st.selectbox("Example perturbation (click a point below, or pick)",
                                  perts, key=f"ex_{ds}")
            pp_ds["sel"] = pp_ds["perturbation"] == chosen
            morder = pp_ds.groupby("method")[metric].mean().sort_values(
                ascending=harder_high).index.tolist()   # best method first
            centers = {m: i for i, m in enumerate(morder)}

            # (a) violin (KDE) per method with raw points jittered inside the envelope
            # (no columnar swarm). Red segment = mean, orange = chosen example. Explicit
            # numeric x (method centres), since Altair's xOffset:Q doesn't fill the band.
            HW = 0.42
            pp_ds["xpos"] = 0.0
            viol = []
            for m in morder:
                mask = pp_ds["method"] == m
                vals = pp_ds.loc[mask, metric].to_numpy()
                g = violin_density(vals)
                g["method"] = m
                g["xL"] = centers[m] - g["dens"] * HW
                g["xR"] = centers[m] + g["dens"] * HW
                viol.append(g)
                pp_ds.loc[mask, "xpos"] = centers[m] + violin_jitter(vals, halfwidth=HW,
                                                                     seed=centers[m])
            viol = pd.concat(viol, ignore_index=True)
            xscale = alt.Scale(domain=[-0.5, len(morder) - 0.5])
            # The labelExpr axis doesn't survive Altair's layer/scale merge, so the
            # method names are drawn as an explicit text layer in a padded band below
            # the data (`ylab`), which always renders.
            yv = pd.concat([viol["y"], pp_ds[metric]]).astype(float)
            ymin, ymax = float(yv.min()), float(yv.max())
            pad = (ymax - ymin) * 0.18 or 0.1
            yscale = alt.Scale(domain=[ymin - pad, ymax], nice=False)
            ylab = ymin - pad * 0.45

            # the single y-axis (with gridlines) lives on this first layer; the other
            # layers set axis=None so they don't override/suppress it on the merge.
            yaxis = alt.Axis(grid=True, gridOpacity=0.4, title=mlabel)
            violin = alt.Chart(viol).mark_area(opacity=0.25, color="#6c8ebf").encode(
                x=alt.X("xL:Q", scale=xscale, axis=None), x2="xR:Q",
                y=alt.Y("y:Q", scale=yscale, axis=yaxis), detail="method:N")
            click_sel = alt.selection_point(name="clickpert", fields=["perturbation"],
                                            on="click", empty=False)
            pts = alt.Chart(pp_ds).mark_circle().encode(
                x=alt.X("xpos:Q", title=None, scale=xscale, axis=None),
                y=alt.Y(f"{metric}:Q", scale=yscale),
                color=alt.Color("sel:N", scale=alt.Scale(domain=[False, True],
                                range=["#9aa0a6", "orange"]), legend=None),
                size=alt.Size("sel:N", scale=alt.Scale(domain=[False, True],
                              range=[26, 170]), legend=None),
                order=alt.Order("sel:N"),
                tooltip=["method", "perturbation", alt.Tooltip(f"{metric}:Q", format=".3f")],
            ).add_params(click_sel)
            means = pp_ds.groupby("method", as_index=False)[metric].mean()
            mean_seg = pd.concat([
                pd.DataFrame({"method": m, "x": [centers[m] - 0.3, centers[m] + 0.3],
                              "val": [v, v]})
                for m, v in zip(means["method"], means[metric])
            ], ignore_index=True)
            mean_layer = alt.Chart(mean_seg).mark_line(color="red", size=2).encode(
                x=alt.X("x:Q", scale=xscale, axis=None),
                y=alt.Y("val:Q", scale=yscale), detail="method:N")
            lab = pd.DataFrame({"x": [centers[m] for m in morder], "y": ylab, "method": morder})
            labels = alt.Chart(lab).mark_text(
                align="center", baseline="middle", fontSize=15, fontWeight="bold",
                color="#333").encode(
                x=alt.X("x:Q", scale=xscale, axis=None),
                y=alt.Y("y:Q", scale=yscale), text="method:N")
            event = st.altair_chart(
                (violin + pts + mean_layer + labels).properties(height=440),
                use_container_width=True, on_select="rerun", key=f"bee_{ds}_{metric}")
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
    pub_paths = [q for q in (_localize(pub_single), _localize(pub_double)) if q]
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
                # label the dataset by perturbation type (norman = double, adamson = single)
                ngenes = [len([x for x in str(p).split("+") if x != "ctrl"]) for p in pert_order]
                ptype = "double" if ngenes and sum(n >= 2 for n in ngenes) > len(ngenes) / 2 else "single"
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
                ).properties(height=300, width=panel_w,
                             title=f"{mlabel} — {ds} ({ptype}-perturbation)")
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
    # Pick the difficulty metric for the y-axis; L2 stays the magnitude control for
    # the partial correlation (centrality correlates with effect magnitude — the
    # confound we found earlier).
    st.header("Gene-network centrality vs difficulty (OmniPath)")
    if centrality is None:
        st.info(f"No `{Path(centrality_path).name}` — run `pixi run -e omnipath "
                "python scripts/gene_centrality.py` to enable this.")
    else:
        c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1.6, 1.4])
        cds = c1.selectbox("dataset", datasets_sel, key="cent_ds")
        ymetric = c2.selectbox("difficulty (y-axis)", list(METRIC_LABELS),
                               format_func=METRIC_LABELS.get, key="cent_ymetric")
        measure = c3.selectbox("centrality measure", CENTRALITY_MEASURES, key="cent_measure")
        # how to combine the two genes of a double: max = "contains a hub", sum = total
        # regulatory load, mean = average. (Singles reduce to the one gene either way.)
        agg = c4.selectbox("combine genes", ["max", "sum", "mean"], key="cent_agg")
        # symlog (not log) for the heavy-tailed centrality axis: log-like but keeps the
        # zeros (out_degree/betweenness = 0). Spearman is rank-based so unaffected.
        logx = c5.checkbox("log centrality", value=True, key="cent_logx")

        ylabel = METRIC_LABELS[ymetric]
        harder_high = ymetric == "l2"   # L2: higher = harder; pearson*: higher = easier
        d = df[(df["dataset"] == cds) & (df["split"] == split)].dropna(subset=[ymetric])
        g = (d.groupby("perturbation")
             .agg(y=(ymetric, "mean"), mag=("l2", "mean")).reset_index())
        g["cent"] = g["perturbation"].map(
            perturbation_centrality(g["perturbation"], centrality, measure, agg))
        g = g.dropna(subset=["cent", "y"])
        if len(g) < 3:
            st.info("Too few perturbations with centrality + difficulty to correlate.")
        else:
            # TODO(revisit): centrality-vs-difficulty correlation stats commented out —
            # the partial-correlation / magnitude-confound interpretation needs review
            # before it's defensible. Plot stays; the stats note is parked here.
            # rk = g[["cent", "y", "mag"]].rank()
            # cm = rk.corr()
            # r_cy, r_cm, r_ym = cm.loc["cent", "y"], cm.loc["cent", "mag"], cm.loc["y", "mag"]
            # if ymetric == "l2":            # y IS the magnitude -> partialling it out is degenerate
            #     partial, ptxt = float("nan"), "n/a (y is magnitude)"
            # else:
            #     den = ((1 - r_cm ** 2) * (1 - r_ym ** 2)) ** 0.5
            #     partial = (r_cy - r_cm * r_ym) / den if den > 0 else float("nan")
            #     ptxt = f"{partial:+.2f}"
            # dir_note = ("higher = harder, so positive ρ ⇒ central genes harder"
            #             if harder_high else
            #             "higher = easier, so negative ρ ⇒ central genes harder")
            # st.caption(
            #     f"Spearman ρ(centrality, {ylabel}) = **{r_cy:+.2f}**  ·  partial ρ(· | L2) = "
            #     f"**{ptxt}**  ·  ρ(centrality, L2) = **{r_cm:+.2f}**  (n={len(g)}). "
            #     f"{ylabel}: {dir_note}; a partial ≈ 0 with strong ρ-vs-L2 ⇒ magnitude confound.")
            xscale = alt.Scale(type="symlog") if logx else alt.Scale()
            enc_x = alt.X("cent:Q", title=f"{measure} ({agg} over the pair's genes)", scale=xscale)
            base_c = alt.Chart(g)
            pts_c = base_c.mark_circle(size=80, opacity=0.6).encode(
                x=enc_x, y=alt.Y("y:Q", title=ylabel, scale=alt.Scale(zero=False)),
                size=alt.Size("mag:Q", title="L2 (magnitude)"),
                tooltip=["perturbation", alt.Tooltip("cent:Q", format=".2f"),
                         alt.Tooltip("y:Q", format=".3f", title=ylabel),
                         alt.Tooltip("mag:Q", format=".2f", title="L2")],
            )
            trend_c = base_c.transform_loess("cent", "y").mark_line(
                color="#e45756", size=2).encode(x=enc_x, y="y:Q")
            k = min(8, len(g))           # label the hardest perturbations
            hard = g.nlargest(k, "y") if harder_high else g.nsmallest(k, "y")
            labels_c = alt.Chart(hard).mark_text(align="left", dx=6, dy=-4, fontSize=10,
                                                 color="#555").encode(
                x=enc_x, y="y:Q", text="perturbation")
            title = f"n = {len(g)}    (red = loess; labels = hardest)"
            st.altair_chart((pts_c + trend_c + labels_c)
                            .properties(height=420, title=title).interactive(),
                            use_container_width=True)
            cov = perturbation_centrality(g["perturbation"], centrality, measure, agg).notna().mean()
            st.caption(f"OmniPath coverage of {cds} perturbations: {cov*100:.0f}%.")
