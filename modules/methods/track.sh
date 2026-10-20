# Sourced by a method run.sh, AFTER $DATA_ROOT, $ds and $seed are set:
#   source "$REPO/modules/methods/track.sh" <method>
# Reads $OMNI_TRACK_CONFIG (default $DATA_ROOT/track.yaml, see track.example.yaml) and
# exports the standard WANDB_* env vars, so any wandb.init() in the job (the vendored fork's
# own hooks, or omni_track.py) picks up project/entity/mode/group/tags/run name unchanged.
# No config file -> nothing exported, OMNI_TRACK=0, the job runs exactly as without it.
_track_cfg="${OMNI_TRACK_CONFIG:-$DATA_ROOT/track.yaml}"
export OMNI_TRACK=0
if [[ -f $_track_cfg ]]; then
  _track_get() {  # flat `key: value`, comments and quotes stripped
    sed -nE "s/^$1:[[:space:]]*([^#]*).*/\1/p" "$_track_cfg" | head -1 | sed -E "s/[[:space:]]+$//; s/^[\"']//; s/[\"']$//"
  }
  _v=$(_track_get mode); export WANDB_MODE="${_v:-online}"
  if [[ $WANDB_MODE != disabled ]]; then
    export OMNI_TRACK=1
    _v=$(_track_get project); export WANDB_PROJECT="${_v:-omni-scfm}"
    _v=$(_track_get entity);  [[ -n $_v ]] && export WANDB_ENTITY="$_v"
    _v=$(_track_get group);   [[ -n $_v ]] && export WANDB_RUN_GROUP="$_v"
    _v=$(_track_get tags)
    _v="${_v// /}"; export WANDB_TAGS="${_v:+$_v,}$1,$ds,seed$seed"
    export WANDB_NAME="$ds-seed$seed-$1"
    # wandb writes run files under the cwd, which is the per-job mktemp sandbox that run.sh
    # deletes on exit; keep them (offline runs live only there until `wandb sync`).
    export WANDB_DIR="${WANDB_DIR:-$DATA_ROOT}"; mkdir -p "$WANDB_DIR/wandb"
    export PYTHONPATH="$(dirname "${BASH_SOURCE[0]}")${PYTHONPATH:+:$PYTHONPATH}"  # omni_track.py
    echo "$1: tracking on (wandb $WANDB_MODE, project $WANDB_PROJECT, run $WANDB_NAME; $_track_cfg)"
  fi
  unset -f _track_get; unset _v
fi
unset _track_cfg
