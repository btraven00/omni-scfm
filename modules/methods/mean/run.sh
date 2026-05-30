#!/usr/bin/env bash
# Mean-prediction baseline — runs the paper's run_mean_prediction.R verbatim via
# the shared R-method runner (handles the working-dir layout, reader swap, gzip).
exec bash modules/methods/run_r_method.sh run_mean_prediction.R "$@"
