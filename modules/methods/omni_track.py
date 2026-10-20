"""In-loop metric logging for the method modules: `import omni_track; omni_track.log(loss=...)`.

A no-op unless modules/methods/track.sh switched tracking on (OMNI_TRACK=1, from track.yaml),
and still a no-op (one warning) if wandb isn't installed in the method's env, so it is safe to
sed into a vendored loop unconditionally. Run identity (project, entity, mode, name, group,
tags) comes from the WANDB_* env vars track.sh exports. wandb.init/log leave the python, numpy
and torch RNG states untouched (checked for scFoundation), so results don't change.
"""
import os
import sys

_run = None
_off = os.environ.get("OMNI_TRACK") != "1"


def _get():
    global _run, _off
    if _run is None and not _off:
        try:
            import wandb
        except ImportError:
            print("omni_track: WARNING tracking on but wandb is not installed in this env", file=sys.stderr)
            _off = True
            return None
        _run = wandb.run or wandb.init()  # reuse a run the vendored code already opened
    return _run


def log(step=None, **metrics):
    run = _get()
    if run is not None:
        run.log(metrics, step=step)


def finish():
    if _run is not None:
        _run.finish()


if __name__ == "__main__":
    # self-check: with tracking off, everything is a silent no-op
    os.environ.pop("OMNI_TRACK", None)
    assert _off and _get() is None
    log(loss=1.0)
    finish()
    print("omni_track: ok (off -> no-op)")
