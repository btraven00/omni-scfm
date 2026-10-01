#!/usr/bin/env bash
# lpm with P from the Replogle RPE1 pseudobulk-PCA embedding.
# One line over the shared runner; see modules/methods/lpm_emb.sh.
exec bash modules/methods/lpm_emb.sh pert replogle_rpe1_essential "$@"
