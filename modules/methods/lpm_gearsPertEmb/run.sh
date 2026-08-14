#!/usr/bin/env bash
# lpm with P from the GEARS' GO-graph spectral embedding.
# One line over the shared runner; see modules/methods/lpm_pert_emb.sh.
exec bash modules/methods/lpm_pert_emb.sh gears_go "$@"
