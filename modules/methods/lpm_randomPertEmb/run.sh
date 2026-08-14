#!/usr/bin/env bash
# lpm with a RANDOM perturbation embedding: P ~ rnorm, one i.i.d. column per condition
# (run_linear_pretrained_model.R:127). A held-out perturbation's column then says nothing
# about which gene was hit, so its prediction is noise — the null the lpm_*PertEmb
# variants are read against. Seed-dependent (rnorm under set.seed(--seed)).
exec bash modules/methods/run_r_method.sh run_linear_pretrained_model.R \
  --pert_embedding random "$@"
