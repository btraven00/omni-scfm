#!/usr/bin/env python3
"""Fetch OmniPath, compute per-gene network centralities, cache a small CSV.

One-time / offline step (needs network + the `omnipath` package). The dashboard
reads the committed CSV (`data/gene_centrality.csv`) — no omnipath dependency at
serve time, so it stays offline/stlite-safe. Output is restricted to the genes we
actually perturb (from scores.parquet) so the artifact is tiny + committable.

OmniPath is a curated, DIRECTED signaling network, so out-degree = downstream
regulatory breadth ("how many targets this gene drives") — a mechanistic proxy
for how broad/non-additive a perturbation's effect is. Unlike GEARS' engineered
k-NN GO graph, it's genuinely heavy-tailed, which is what the centrality-vs-
difficulty analysis needs.

Usage:
  python scripts/gene_centrality.py [--scores out/scores.parquet]
                                    [--out data/gene_centrality.csv]
                                    [--betweenness-k 800] [--all-genes]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import pandas as pd


def _genes_from_conditions(conditions) -> set[str]:
    genes: set[str] = set()
    for p in conditions:
        genes.update(g for g in str(p).split("+") if g and g != "ctrl")
    return genes


def perturbed_genes(scores_path: str) -> set[str]:
    df = pd.read_parquet(scores_path)
    return _genes_from_conditions(df["perturbation"].dropna().unique())


def fetch_network():
    """OmniPath directed interaction graph (source -> target gene symbols)."""
    import omnipath as op
    # genesymbols=True adds source_genesymbol/target_genesymbol (default ids are UniProt).
    net = op.interactions.AllInteractions().get(genesymbols=True)
    return nx.from_pandas_edgelist(net, "source_genesymbol", "target_genesymbol",
                                   create_using=nx.DiGraph())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="out/scores.parquet")
    ap.add_argument("--genes-file", default=None,
                    help="newline-delimited gene or condition list (instead of --scores)")
    ap.add_argument("--out", default="data/gene_centrality.csv")
    ap.add_argument("--betweenness-k", type=int, default=800,
                    help="node samples for approximate betweenness (0 = skip)")
    ap.add_argument("--all-genes", action="store_true",
                    help="emit every graph gene, not just our perturbed genes")
    args = ap.parse_args()

    G = fetch_network()
    print(f"OmniPath: {G.number_of_nodes()} genes, {G.number_of_edges()} edges", file=sys.stderr)

    outd, ind, totd = dict(G.out_degree()), dict(G.in_degree()), dict(G.degree())
    pr = nx.pagerank(G)
    btw = ({} if not args.betweenness_k
           else nx.betweenness_centrality(G, k=min(args.betweenness_k, G.number_of_nodes()),
                                          seed=0))

    if args.all_genes:
        genes = set(G.nodes())
    elif args.genes_file:
        lines = Path(args.genes_file).read_text().split()
        genes = _genes_from_conditions(lines)
    else:
        genes = perturbed_genes(args.scores)
    nan = float("nan")
    rows = [{
        "gene": g, "in_omnipath": g in G,
        "out_degree": outd.get(g, nan), "in_degree": ind.get(g, nan),
        "degree": totd.get(g, nan), "pagerank": pr.get(g, nan),
        "betweenness": btw.get(g, nan) if btw else nan,
    } for g in sorted(genes)]
    out = pd.DataFrame(rows)

    import omnipath
    cov = out["in_omnipath"].mean() if len(out) else 0.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf8") as fh:
        fh.write(f"# omnipath={getattr(omnipath, '__version__', '?')} "
                 f"graph={G.number_of_nodes()}n/{G.number_of_edges()}e "
                 f"genes={len(out)} in_omnipath={int(out['in_omnipath'].sum())}\n")
    out.to_csv(args.out, mode="a", index=False)
    print(f"wrote {args.out}: {len(out)} genes, {cov*100:.0f}% in OmniPath", file=sys.stderr)


if __name__ == "__main__":
    main()
