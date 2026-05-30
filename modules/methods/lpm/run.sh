#!/usr/bin/env bash
# Linear pretrained model, self-trained (lpm_selftrained) — the paper's headline
# linear baseline. Runs run_linear_pretrained_model.R verbatim via the shared
# R-method runner. Defaults give gene/pert embeddings from the training data
# (pca_dim=10, ridge_penalty=0.1), so no external embedding files are needed.
exec bash modules/methods/run_r_method.sh run_linear_pretrained_model.R "$@"
