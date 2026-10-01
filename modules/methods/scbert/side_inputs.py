"""Build scBERT's two side inputs for one job, from the checkpoint + the committed gene list.

    python side_inputs.py <panglao_pretrain.pth> <genes.txt> <out_h5ad> <out_npy>

1. panglao_human.h5ad — run_scbert.py reads only its gene axis (var_names, X.shape[1],
   var.copy()) to lay the data out in scBERT's 16906-gene order. The real file is 11 GB
   (1,357,593 cells); its var holds only _index. A 0-cell AnnData with the same var_names
   is equivalent for everything the script touches. Gene list = that h5ad's var_names,
   identical (same order) to HF kaichenxu/cape_scbert's vocab.json.
2. gene2vec_16906.npy — performer_pytorch.py:442 loads it from <cwd>/../data/. It is the
   checkpoint's pos_emb.emb.weight minus its zero padding row: verified BYTE-IDENTICAL to
   the copy in the author's data.zip (Drive 1ZyC2mGxZK1tLdy3sMOmdJERnSuvR9WZ0), so
   rebuilding it here reproduces the original file exactly.

The checkpoint is loaded with weights_only=True: its pickle references only torch tensor
types (checked), so no code can run from it.
"""
import hashlib
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

GENES_SHA256 = "4c09ba1912e184682fe5e7f32e2b443bdfb7883d0d0191551b38c6c2f155600f"  # "\n".join(genes)
GENE2VEC_SHA256 = "fbf26c229cf7d6c01bc0449fde4b0bb85e558ae53c5ff19b27eaa9ae5447604d"  # the author's npy


def main(ckpt, genes_txt, out_h5ad, out_npy):
    genes = open(genes_txt).read().split()
    if hashlib.sha256("\n".join(genes).encode()).hexdigest() != GENES_SHA256:
        sys.exit(f"side_inputs: {genes_txt} is not the scBERT panglao gene list")
    ad.AnnData(sp.csr_matrix((0, len(genes)), dtype=np.float32),
               var=pd.DataFrame(index=pd.Index(genes))).write_h5ad(out_h5ad)

    w = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"]["pos_emb.emb.weight"]
    np.save(out_npy, np.ascontiguousarray(w.numpy()[:-1]))
    if hashlib.sha256(open(out_npy, "rb").read()).hexdigest() != GENE2VEC_SHA256:
        sys.exit("side_inputs: rebuilt gene2vec_16906.npy differs from the author's file")


if __name__ == "__main__":
    main(*sys.argv[1:5])
