#!/usr/bin/env bash
# lpm with a RANDOM gene embedding: G ~ rnorm (genes x pca_dim), confining predictions to
# a random subspace of gene space (run_linear_pretrained_model.R:105). The null the
# lpm_scgptGeneEmb / lpm_scFoundationGeneEmb variants are read against.
exec bash modules/methods/run_r_method.sh run_linear_pretrained_model.R \
  --gene_embedding random "$@"
