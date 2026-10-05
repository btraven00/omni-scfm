"""Patch a COPY of the forked GEARS so its per-epoch train-set evaluation is skipped,
bit-identically. Used by run.sh when OMNI_SCF_SKIP_TRAIN_EVAL=1.

    python skip_train_eval.py <fork>/gears/gears.py              # skip (+ trace if asked)
    python skip_train_eval.py <fork>/gears/gears.py --trace-only # keep the pass, trace only

Why it is safe: after each epoch GEARS.train() runs evaluate() over the WHOLE train loader
(gears.py:410). The result (train_res -> train_metrics) is only printed and optionally
logged to wandb; best-model selection uses val_metrics only. The pass costs ~40% of an
scFoundation run (>5 h per epoch on norman_from_scfoundation).

Why it is not simply deleted: the train loader is shuffle=True with no generator of its
own, so iterating it draws from torch's GLOBAL RNG and fixes every later epoch's shuffle
order. Verified (torch 2.0.1 and 2.7.1, torch_geometric DataLoader): fetching just the
FIRST batch leaves the global RNG in exactly the state a full pass does (the sampler draws
its seed on the first batch; creating the iterator alone is not enough), and the next
epoch's batches come out identical. eval-mode forwards draw no random numbers. So the
patch replaces the pass by next(iter(train_loader)) and reports the train metrics as NaN.

OMNI_SCF_RNG_TRACE=1 additionally prints a hash of the CPU and CUDA RNG states at that
point every epoch — for proving the skip is bit-identical (compare with a full run).
"""
import sys

path = sys.argv[1]
trace_only = "--trace-only" in sys.argv[2:]
src = open(path).read()

EVAL = ("            train_res = evaluate(train_loader, self.model, "
        "self.config['uncertainty'], self.device)\n")
METRICS = "            train_metrics, _ = compute_metrics(train_res)\n"
assert src.count(EVAL) == 1 and src.count(METRICS) == 1, "gears.py changed: patch anchors not found"

trace = '''            if os.environ.get("OMNI_SCF_RNG_TRACE") == "1":
                import hashlib
                _h = lambda t: hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16]
                _c = "|".join(_h(s) for s in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else "-"
                print_sys(f"RNG-TRACE epoch {epoch + 1}: cpu {_h(torch.get_rng_state())} cuda {_c}")
'''
if trace_only:   # baseline for the proof: the real pass, then the same RNG trace
    src = src.replace(EVAL, EVAL + trace)
else:
    src = src.replace(EVAL, '''            # omni-scfm (OMNI_SCF_SKIP_TRAIN_EVAL): skip the print-only train-set pass but make
            # the same global-RNG draws it makes (= fetching its first batch). See
            # modules/methods/scfoundation/skip_train_eval.py.
            next(iter(train_loader))
            train_res = None
''' + trace)
    src = src.replace(METRICS, '''            from collections import defaultdict
            train_metrics = defaultdict(lambda: float("nan"))   # skipped: printed as nan
''')
open(path, "w").write(src)
print(f"skip_train_eval: patched {path} ({'trace only' if trace_only else 'skip train-set eval'})")
