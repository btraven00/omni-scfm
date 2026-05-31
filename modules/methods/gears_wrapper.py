#!/usr/bin/env python3
"""Run a vendored paper python script VERBATIM, with minimal compat shims.

WHY THIS EXISTS — the cell-gears 0.1.2 / modern-scipy incompatibility
---------------------------------------------------------------------
cell-gears 0.1.2 (the GEARS version the paper used) was written against an older
scientific stack. In `GEARS.__init__` (gears/gears.py:87) it does:

    np.mean(self.adata.X[self.adata.obs.condition == 'ctrl'], axis=0)

i.e. it indexes a **scipy sparse matrix** with a **boolean pandas Series**. As of
scipy's sparse-indexing rewrite, `scipy.sparse._index._validate_indices` calls
`ix.nonzero()` on that index — but pandas removed `Series.nonzero` back in pandas
1.0. So on a current stack the model fails at init with:

    AttributeError: 'Series' object has no attribute 'nonzero'

This is purely a library-version interaction, NOT a bug in our wiring or in the
vendored script (GEARS otherwise trains/predicts fine — verified end-to-end).

OBSERVED in the `gears-gpu` env (envs/gears-gpu.yml): python 3.10.20,
cell-gears 0.1.2, torch 2.3.1+cu, numpy 1.26.4, pandas 2.3.3, scipy 1.15.2.
(The older CPU `gears` env never hit this because only GEARS — not additive —
exercises that line.)

FIX: we don't pin the whole stack back down (that fights scipy/pyg resolution and
loses the easy CUDA torch wheels); GEARS is a *stochastic* trained model, so a
newer scipy in this one mean-over-controls is numerically irrelevant. Instead we
restore the single removed method, then run the script unchanged — analogous to
wrapper.R neutralising one line. If a future stack bump breaks another call,
extend the shims here rather than re-pinning.

The vendored script path arrives via $OMNI_VENDORED_SCRIPT so its own --flags are
the only things on argv (argparse there reads sys.argv as usual).
"""
import os
import runpy
import sys

import numpy as np
import pandas as pd

# Restore the API scipy's sparse boolean indexing still expects from the index.
if not hasattr(pd.Series, "nonzero"):
    pd.Series.nonzero = lambda self: (np.asarray(self).nonzero())

script = os.environ.get("OMNI_VENDORED_SCRIPT")
if not script:
    raise SystemExit("OMNI_VENDORED_SCRIPT is not set")
runpy.run_path(script, run_name="__main__")
