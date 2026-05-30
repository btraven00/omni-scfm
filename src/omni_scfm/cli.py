"""Helpers for OmniBenchmark module entrypoints.

OmniBenchmark invokes a module as::

    <interp> <entrypoint> --output_dir DIR --name ID --<input_id> PATH [--param V]

and, for gather/collector modules, repeats values for one flag::

    ... --predictions f1 f2 f3 ...

`parse_ob_args` turns that into a dict. Input ids may contain dots
(e.g. ``data.raw``); look them up by their exact id.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def parse_ob_args(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse ``--key value [value ...]`` tokens into a dict.

    A flag with one value maps to a ``str``; with several to a ``list[str]``;
    with none to ``True``. Keys keep their original spelling (dots included).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    out: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:]
        i += 1
        vals: list[str] = []
        while i < len(argv) and not argv[i].startswith("--"):
            vals.append(argv[i])
            i += 1
        out[key] = True if not vals else (vals[0] if len(vals) == 1 else vals)
    return out


def require(args: dict[str, Any], *keys: str) -> Any:
    """Return the first present key's value, or raise with a clear message."""
    for k in keys:
        if k in args:
            return args[k]
    raise SystemExit(f"missing required argument (one of): {', '.join('--' + k for k in keys)}")


def dataset_name(args: dict[str, Any], *input_keys: str) -> str:
    """Resolve the ``{dataset}`` wildcard for naming outputs.

    OmniBenchmark names a stage's outputs ``{dataset}.<ext>`` where ``{dataset}``
    propagates from the dataset-stage module id. Rather than depend on ``--name``
    (which is the dataset id under API <= 0.4 but the *module* id under >= 0.5),
    derive it from an input file's basename (the upstream output is already
    ``{dataset}.<ext>``). Falls back to ``--name``.
    """
    for k in input_keys:
        v = args.get(k)
        if isinstance(v, str):
            return Path(v).name.split(".")[0]
        if isinstance(v, list) and v:
            return Path(v[0]).name.split(".")[0]
    return str(args.get("name", "dataset"))
