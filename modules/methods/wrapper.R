#!/usr/bin/env Rscript
# Run a vendored paper R script VERBATIM, neutralising its single
# machine-specific line.
#
# run_mean_prediction.R (and siblings) hardcode
#   Sys.setenv("BASILISK_EXTERNAL_CONDA" = "/g/easybuild/.../Miniforge3/...")
# a cluster path that does not exist elsewhere and breaks zellkonverter's
# basilisk-backed h5ad reader. We cannot edit the pinned submodule, so we mask
# Sys.setenv for that one key: the sourced script's unqualified `Sys.setenv(...)`
# resolves to this global definition and the bad key is dropped, leaving
# basilisk to use its own managed environment. Every other line runs unchanged.
#
# The vendored script path arrives via $OMNI_VENDORED_SCRIPT so that the script's
# own --flags stay the only entries on commandArgs() for argparser.

.orig_setenv <- base::Sys.setenv
assign("Sys.setenv", function(...) {
  a <- list(...)
  a[["BASILISK_EXTERNAL_CONDA"]] <- NULL
  if (length(a)) do.call(.orig_setenv, a) else invisible(TRUE)
}, envir = globalenv())

# Read .h5ad via picklerick (native Rust->R, no basilisk/Python). The scripts call
# `zellkonverter::readH5AD(file)`; we replace that in the zellkonverter namespace
# so they run verbatim but read through picklerick. picklerick names the X assay
# "counts"; rename to "X" to match what zellkonverter returns (scripts use
# assay(sce, "X")). zellkonverter is kept installed only so the `::` call resolves.
if (requireNamespace("picklerick", quietly = TRUE) &&
    requireNamespace("zellkonverter", quietly = TRUE)) {
  .pr_readH5AD <- function(file, ...) {
    sce <- picklerick::read_h5ad(as.character(file), as = "SingleCellExperiment")
    an <- SummarizedExperiment::assayNames(sce)
    if ("counts" %in% an && !("X" %in% an)) {
      SummarizedExperiment::assayNames(sce)[an == "counts"] <- "X"
    }
    sce
  }
  utils::assignInNamespace("readH5AD", .pr_readH5AD, ns = "zellkonverter")
  message("wrapper.R: zellkonverter::readH5AD -> picklerick::read_h5ad")
}

script <- Sys.getenv("OMNI_VENDORED_SCRIPT")
if (!nzchar(script)) stop("OMNI_VENDORED_SCRIPT is not set")
source(script)
