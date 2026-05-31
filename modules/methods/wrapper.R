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

# TODO -- we can get rid of all the basilisk/zellkonverter stuff now no?
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

# run_mean_prediction.R builds `pred` = replicate(nrow(psce), mean_vec) — one entry
# per *gene* (nrow=genes), but names(pred) <- clean_condition only labels the first
# n_condition entries (R pads the rest with NA). The duplicate / NA-named entries are
# collapsed later by run_r_method.sh's python json.load. On a wide dataset
# (norman_from_scfoundation: 19264 genes -> 19264 x 19264 numbers ~7.4GB) that single
# rjson string blows past R's 2^31-1 byte limit before python ever sees it; the
# GEARS-filtered datasets (~5060 genes -> ~0.5GB) stay under it. We override
# rjson::toJSON to detect the over-limit case and collapse duplicate keys up front
# (first-occurrence order, last value — what python's json.load yields), so the
# emitted JSON is identical post-collapse. Under-limit inputs pass straight through to
# the original, leaving every other dataset/method byte-identical.
if (requireNamespace("rjson", quietly = TRUE)) {
  .orig_toJSON <- rjson::toJSON
  .dedupe_toJSON <- function(x) {
    keys <- ifelse(is.na(names(x)), "NA", names(x))
    uniq <- keys[!duplicated(keys)]            # python dict key order (first occurrence)
    last <- tapply(seq_along(keys), keys, max)  # python last-write-wins value per key
    parts <- vapply(uniq, function(k)
      paste0(.orig_toJSON(k), ":", .orig_toJSON(x[[last[[k]]]])), character(1))
    paste0("{", paste(parts, collapse = ","), "}")
  }
  .guarded_toJSON <- function(x, ...) {
    if (is.list(x) && length(x) > 1L && !is.null(names(x)) &&
        all(vapply(x, is.numeric, logical(1)))) {
      approx <- as.numeric(length(x)) * (nchar(.orig_toJSON(x[[1]])) + nchar(names(x)[1]) + 4)
      if (approx > 2^31 * 0.95) {
        message("wrapper.R: rjson::toJSON would exceed R's 2^31 byte limit; ",
                "collapsing duplicate keys before serialising")
        return(.dedupe_toJSON(x))
      }
    }
    .orig_toJSON(x, ...)
  }
  utils::assignInNamespace("toJSON", .guarded_toJSON, ns = "rjson")
}

script <- Sys.getenv("OMNI_VENDORED_SCRIPT")
if (!nzchar(script)) stop("OMNI_VENDORED_SCRIPT is not set")
source(script)
