#!/bin/bash
# Run one campaign item (one line of a repro/ item list) into <out_root>/<key>/.
# Usage: bash scripts/run_item.sh <out_root> <item line words...>
# Item forms:
#   <suite> <perturbation> <task_id> <trial_offset> <n_trials> <arm>     LIBERO-PRO
#     arm tokens: wv --wrist-verify, rb --relation-bind, _or SPARK_ORACLE_RETRY=1 (the success check after
#     an attempt decides the retry), _ph SPARK_PHRASES_V2=1, _rg SPARK_PICK_REGROUND=1, _rgs also
#     SPARK_PICK_REGROUND_GATE_M=0.05. repro/libero_pro_table1_items.txt runs wvrb_or_ph_rgs; the strict
#     list runs wvrb_ph_rgs.
#     _pc --post-conditions (also on displace arms; SPARK_DISPLACE_MAGS narrows the displacements);
#     _skgen, _skbase, _skplain, _skshape, _skask, _skaskw: SPARK_SKILLSET=generic, base, plain, shape, ask, askw
#     (spark_core/skillset.py).
#     _mf SPARK_MOTION_FIX=1 (libero_pro/motion.py: reach check, opposite-rim retry, failed pick ends its sequence).
#     _p2 SPARK_PROMPT=v2: the consolidated planner prompt (prompts/system_franka_v2.txt).
#     Recovery after a failed attempt: default retract + re-plan (blind replan); _lr --layered-recovery (the
#     ladder: re-bind, local retry, replan); _norec --no-blind-replan (none).
#     _dev --dev-placements: re-sampled placements, for development seeds (trial_offset >= 1000).
#     Perception noise: _dn2 (depth sigma 0.02 z), _dd10 (10 % depth dropout), _seg4 (masks +-4 px), _cal1 (1 cm,
#     0.5 deg extrinsic error), _nowrist (agentview only), _noagent (wrist only); see the case below.
#   displace <suite> <task_id> <arm>                                     fixed-plan displacement
#     arm on|onec|sd|lr|off: on --scene-diff --layered-recovery, onec also --event-captures, sd or lr
#     only that one mechanism, off neither;
#     llm- prefix: the planner writes the plan (llm-on, llm-sd, llm-lr, llm-off, llm-replan);
#     suffix _st: strict (SPARK_NO_ORACLE=1, only the robot's own checks start a recovery);
#     suffix _inst: instantaneous displacement (one state write) instead of the 40-step slide;
#     tokens:
#       _mf SPARK_MOTION_FIX=1
#       _fr SPARK_FOLLOW_RETARGET=1 (the descent follows a mid-motion retarget)
#       _ip SPARK_PICK_INSTANCE_CHECK=1 (with _mf: a failed pick keeps its instance only while something stands there)
#       _sdx --scene-diff on any arm (llm-replan_sdx: replan with scene diff)
#       _rf --rebind-fresh
#       _ud --scene-diff-unknown
#       _vs --validator-object-snap
#       _fs --freshness-steps (7 steps stand for the 50 ms at about 7 ms a step; SPARK_FRESHNESS_STEPS overrides)
#     SPARK_PLAN_REPLAY=<root> replays the plans recorded in
#     <root>/displace_<suite>_<task>_<arm without those tokens>/dyn.json, with no LLM call;
#     SPARK_TRACE=1 saves a per-step trace of each trial in <key>/trace/ (dyn/trace.py), and
#     SPARK_TRACE_FRAMES=N also the rendered images every N env steps.
# The item directory is the claim (atomic mkdir), so any number of these may run at once, on one or
# many machines sharing <out_root>. An item that finishes without a valid result is moved to
# <out_root>/failed/ and runs again on the next pass.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
OUT=${1:?out_root}; shift
KEY=$(echo "$*" | tr ' :' '__')
DIR=$OUT/$KEY
mkdir -p "$OUT"
mkdir "$DIR" 2>/dev/null || exit 0   # done or claimed elsewhere
# interrupted: release the claim so the next pass runs the item again
trap 'mkdir -p "$OUT/failed"; mv "$DIR" "$OUT/failed/$KEY.$(date +%s)"; exit 130' INT TERM
LOG=$DIR/log.txt
cd "$SPARK_ROOT"

case "$1" in
  displace)
    # Scripted plans (no planner); 3 injection phases x 0/2/5/10 cm. Without _st the executor's
    # goal-satisfied check and the recovery gate read the simulator predicate.
    read -r _ suite task arm <<< "$*"
    full_arm=$arm
    # llm-<arm>: the planner writes the plan. llm-replan keeps only its retract, re-plan, re-execute;
    # every other llm arm turns that off, so it varies only the named mechanisms.
    plan=(--no-gemini); case "$arm" in llm-replan*) plan=();; llm-*) plan=(--no-blind-replan);; esac
    arm=${arm#llm-}
    extra=(); case "$arm" in on*) extra=(--scene-diff --layered-recovery);; sd*) extra=(--scene-diff);; lr*) extra=(--layered-recovery);; esac
    case "$arm" in onec*) extra+=(--event-captures);; esac
    orc=0; case "$arm" in *_st*) orc=1;; esac
    slide=40; case "$arm" in *_inst*) slide=0;; esac
    case "$arm" in *_rf*) extra+=(--rebind-fresh);; esac; case "$arm" in *_ud*) extra+=(--scene-diff-unknown);; esac
    case "$arm" in *_vs*) extra+=(--validator-object-snap);; esac
    case "$arm" in *_pc*) extra+=(--post-conditions);; esac
    case "$arm" in *_fs*) extra+=(--freshness-steps "${SPARK_FRESHNESS_STEPS:-7}");; esac
    case "$arm" in *_sdx*) extra+=(--scene-diff);; esac   # scene diff added to an arm that does not start with sd/on (llm-replan_sdx)
    mf=(); case "$arm" in *_mf*) mf=(SPARK_MOTION_FIX=1);; esac
    case "$arm" in *_fr*) mf+=(SPARK_FOLLOW_RETARGET=1);; esac
    case "$arm" in *_ip*) mf+=(SPARK_PICK_INSTANCE_CHECK=1);; esac
    if [ -n "${SPARK_PLAN_REPLAY:-}" ]; then
      rec=$full_arm; for tok in _rf _ud _vs _fs _pc _mf _fr _ip; do rec=${rec//$tok/}; done
      extra+=(--plan-replay "$SPARK_PLAN_REPLAY/displace_${suite}_${task}_$rec/dyn.json")
    fi
    [ "${SPARK_TRACE:-0}" = 1 ] && extra+=(--trace-dir "$DIR/trace" --trace-frames "${SPARK_TRACE_FRAMES:-0}")
    # _t5: trials 5-9 (fresh init states) instead of 0-4; _m510: only the 5 and 10 cm displacements.
    toff=0; case "$arm" in *_t5*) toff=5;; esac
    mags=${SPARK_DISPLACE_MAGS:-0,2,5,10}; case "$arm" in *_m510*) mags=5,10;; *_m10*) mags=10;; esac
    # _dv: development check, init states 40-41, 2 trials.
    ntr=5; case "$arm" in *_dv*) toff=40; ntr=2;; esac
    env -u SPARK_MOTION_FIX -u SPARK_FOLLOW_RETARGET -u SPARK_PICK_INSTANCE_CHECK SPARK_NO_ORACLE=$orc "${mf[@]}" python -m spark_bench.dyn.run --suite "$suite" --task-id "$task" "${plan[@]}" --num-trials $ntr \
      --trial-offset $toff --phases post_plan,mid_approach,post_grasp_plan --magnitudes-cm "$mags" \
      --slide-steps "$slide" --dyn-output "$DIR/dyn.json" "${extra[@]}" > "$LOG" 2>&1
    RESULT=$DIR/dyn.json;;
  *)
    read -r suite pert task off n arm <<< "$*"
    extra=(); case "$arm" in *wv*) extra+=(--wrist-verify);; esac; case "$arm" in *rb*) extra+=(--relation-bind);; esac
    case "$arm" in *_pc*) extra+=(--post-conditions);; esac
    case "$arm" in *_lr*) extra+=(--layered-recovery);; *_norec*) extra+=(--no-blind-replan);; esac
    case "$arm" in *_dev*) extra+=(--dev-placements);; esac
    [ "${SPARK_TRACE:-0}" = 1 ] && extra+=(--trace-dir "$DIR/trace" --trace-frames "${SPARK_TRACE_FRAMES:-0}")
    # Perception noise (dev runs only; the runner refuses it without _dev): _dnN depth sigma = 0.0N * z,
    # _ddN N % depth dropout, _segN mask erosion/dilation by N px, _calN extrinsic error N cm (_cal0p5 = 0.5),
    # _nowrist agentview only, _noagent wrist only.
    [[ $arm =~ _dn([0-9]+) ]] && extra+=(--depth-noise-k "0.$(printf %02d "${BASH_REMATCH[1]}")")
    [[ $arm =~ _dd([0-9]+) ]] && extra+=(--depth-dropout "0.$(printf %02d "${BASH_REMATCH[1]}")")
    [[ $arm =~ _seg([0-9]+) ]] && extra+=(--seg-noise-px "${BASH_REMATCH[1]}")
    [[ $arm =~ _cal([0-9p]+) ]] && extra+=(--calib-noise-cm "${BASH_REMATCH[1]//p/.}")
    case "$arm" in *_nowrist*) extra+=(--views agentview);; *_noagent*) extra+=(--views wrist);; esac
    ienv=(); case "$arm" in *_or*) ienv+=(SPARK_ORACLE_RETRY=1);; esac; case "$arm" in *_ph*) ienv+=(SPARK_PHRASES_V2=1);; esac
    case "$arm" in *_rg*) ienv+=(SPARK_PICK_REGROUND=1);; esac; case "$arm" in *_rgs*) ienv+=(SPARK_PICK_REGROUND_GATE_M=0.05);; esac
    case "$arm" in *_p2*) ienv+=(SPARK_PROMPT=v2);; esac
    case "$arm" in *_skgen*) ienv+=(SPARK_SKILLSET=generic);; *_skbase*) ienv+=(SPARK_SKILLSET=base);; *_skplain*) ienv+=(SPARK_SKILLSET=plain);;
      *_skshape*) ienv+=(SPARK_SKILLSET=shape);; *_skaskw*) ienv+=(SPARK_SKILLSET=askw);; *_skask*) ienv+=(SPARK_SKILLSET=ask);; esac
    case "$arm" in *_mf*) ienv+=(SPARK_MOTION_FIX=1);; esac
    case "$arm" in *_ip*) ienv+=(SPARK_PICK_INSTANCE_CHECK=1);; esac
    # The arm alone sets these flags: a value exported in the caller's shell must not leak in.
    env -u SPARK_ORACLE_RETRY -u SPARK_PHRASES_V2 -u SPARK_PICK_REGROUND -u SPARK_PICK_REGROUND_GATE_M -u SPARK_SKILLSET -u SPARK_MOTION_FIX -u SPARK_PICK_INSTANCE_CHECK -u SPARK_PROMPT "${ienv[@]}" \
      bash src/spark_bench/run_headline_libero_pro.sh "$suite" "$pert" "$n" --only-task-id "$task" \
      --trial-offset "$off" "${extra[@]}" --output-dir "$DIR" > "$LOG" 2>&1
    RESULT=$DIR/$suite.json;;
esac

# Not a result: no output file, a GPU memory failure anywhere in the log, a planner error, or a
# task-level "error" (the env failed to build). A trial that ran out of steps is a failed trial, not this.
ok=1
[ -s "$RESULT" ] || ok=0
grep -qE "CUDA OOM|CUDA error|out of memory|\[planner error\]" "$LOG" && ok=0
# A displacement trial stores its exception in dyn.json, not the log; an out-of-memory there is not a failed trial.
if [ $ok = 1 ] && [ "$1" = displace ]; then
  python - "$RESULT" <<'PY' || ok=0
import json, sys
cells = json.load(open(sys.argv[1])).get("cells", [])
sys.exit(any("error" in t for c in cells for t in c["trials"]))
PY
fi
# LIBERO-PRO: a task-level error, a trial exception other than the episode clock running out (that one is
# a failed trial), or fewer trials than asked invalidate the item.
if [ $ok = 1 ] && [ "$1" != displace ]; then
  python - "$RESULT" "$n" <<'PY' || ok=0
import json, sys
r = json.load(open(sys.argv[1]))
tds = [td for v in r.get("perturbations", {}).values() for td in v["task_details"]]
bad = (not tds or any("error" in td or len(td.get("trials", [])) != int(sys.argv[2])
                      or any("error" in tm and "terminated episode" not in tm["error"] for tm in td.get("trial_meta", []))
                      for td in tds))
sys.exit(bad)
PY
fi
if [ $ok = 0 ]; then
  mkdir -p "$OUT/failed"
  mv "$DIR" "$OUT/failed/$KEY.$(date +%s)"
  echo "[item] FAILED $KEY (log in $OUT/failed/)" >&2
  exit 1
fi
echo "[item] done $KEY"
