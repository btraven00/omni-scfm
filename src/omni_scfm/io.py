"""I/O helpers for method outputs (predictions) and gene names."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def _dump_gz(obj, path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf8") as fh:
        json.dump(obj, fh)


def write_predictions(
    output_dir: str | Path,
    dataset: str,
    predictions: dict[str, list[float]],
    gene_names: list[str],
    se: dict[str, list[float]] | None = None,
) -> None:
    """Write a method's per-condition predictions to the OB output dir.

    The large condition x gene tables are gzipped (they compress extremely well —
    especially constant/redundant predictions); gene names stay plain (small).
      - {dataset}.predictions.json.gz      {condition: [per-gene value]}
      - {dataset}.gene_names.json          gene order matching the value vectors
      - {dataset}.predictions_se.json.gz   (optional) per-condition standard errors
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _dump_gz(predictions, out / f"{dataset}.predictions.json.gz")
    with open(out / f"{dataset}.gene_names.json", "w", encoding="utf8") as fh:
        json.dump(list(gene_names), fh)
    if se is not None:
        _dump_gz(se, out / f"{dataset}.predictions_se.json.gz")
