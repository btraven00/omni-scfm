#!/usr/bin/env bash
# lpm with G from scgpt's gene embedding (modules/godata/gene_embedding).
# One line over the shared runner; see modules/methods/lpm_emb.sh.
exec bash modules/methods/lpm_emb.sh gene scgpt_gene "$@"
