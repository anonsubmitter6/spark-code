# Setup

## Requirements

- Tested on Ubuntu 22.04 and 24.04 (x86_64) with an NVIDIA driver for CUDA 12.6, git and [uv](https://docs.astral.sh/uv/).
- Headless rendering through EGL: `sudo apt install libegl1 libgl1 libglvnd0`. The NVIDIA driver
  must provide its EGL vendor file (`10_nvidia.json` in glvnd's `egl_vendor.d`).
- Disk: Qwen weights about 16 GB, SAM3 checkpoint about 3.5 GB, LIBERO-PRO assets about 0.6 GB.

## Install

```bash
git submodule update --init   # if you cloned without --recurse-submodules
bash scripts/setup.sh
hf auth login
```

`setup.sh` applies `scripts/sam3.patch` to the SAM3 submodule, runs `uv sync --extra bench --extra dev`
(CUDA torch from the PyTorch index, everything else from `uv.lock`), downloads the LIBERO-PRO
perturbed suites, checks their fingerprint, writes `.libero/config.yaml` and runs an import check.
It never writes the per-user LIBERO config. Re-running it is safe.

The SAM3 checkpoint is gated: accept the license at https://huggingface.co/facebook/sam3 before
`hf auth login`. `smoke.sh` downloads it and says so if access is missing.

## Planner

```bash
bash scripts/serve_planner.sh          # foreground; add & or use a second terminal
```

vLLM runs in its own cached uv environment (it pins its own torch). The first run downloads the
model. Set `VLLM_MEM` (GPU memory fraction) and `MAX_SEQS` for your GPU, `PORT` for another port.
To use a server on another machine, start it there with `bash scripts/serve_planner.sh --host
0.0.0.0` (it binds 127.0.0.1 by default; the last flag wins) or use an SSH tunnel, and export
`OPENAI_BASE_URL=http://<host>:8000/v1` before sourcing `scripts/env.sh`. To use Gemini instead: `uv sync --extra bench --extra gemini`, `export
SPARK_LLM_BACKEND=gemini GEMINI_API_KEY=...` (not the reported protocol).

## Check

```bash
bash scripts/smoke.sh
```

It runs the unit tests, the import and data checks, an EGL context, the SAM3 checkpoint download,
one trial with a scripted plan, one planner call, two LIBERO-PRO trials (`--wrist-verify
--relation-bind` only, not the full Table 1 arm). It passes when every step runs and writes well-formed results; the success counts it prints
are for information only.

## Reproduce

See [reproduce.md](reproduce.md).

## Environment

Every script sources `scripts/env.sh`, which also activates `.venv`. Source it yourself before
running `python -m spark_bench...` by hand. It puts `third_party/LIBERO` on `PYTHONPATH` (that is how the
runs imported `libero`) and sets `PYTHONNOUSERSITE=1`, `LIBERO_CONFIG_PATH=<repo>/.libero`, `MUJOCO_GL=egl`,
`PYOPENGL_PLATFORM=egl` and the planner variables (`SPARK_LLM_BACKEND`, `SPARK_LLM_MODEL`,
`OPENAI_BASE_URL`, `OPENAI_API_KEY`). Any of them can be exported first to override.

## Environment switches

Unset means the default. The simulation switches select the arm of a run; `scripts/run_item.sh` sets
them from the arm tokens and clears any value exported in the caller's shell.

Simulation (`spark_bench`, `spark_core`, `scripts/`):

| Variable | Default | Effect |
|---|---|---|
| `SPARK_MOTION_FIX` | off | `1` (token `_mf`): reach check before the close, opposite-rim retry, no joint-space fallback on a contact stall, a failed pick fails its leaf |
| `SPARK_PICK_INSTANCE_CHECK` | off | `1` (token `_ip`, with `_mf`): after a failed pick the label stays on its instance only while something still stands there |
| `SPARK_FOLLOW_RETARGET` | off | `1` (token `_fr`): the descent after a monitored move follows a mid-motion retarget |
| `SPARK_SKILLSET` | unset | `generic`, `base`, `plain`, `shape`, `ask` or `askw` (tokens `_sk*`): which leaf skills the planner sees and the executor runs |
| `SPARK_NO_ORACLE` | off | `1` (token `_st`): no control decision reads the simulator's success predicate |
| `SPARK_ORACLE_RETRY` | off | `1` (token `_or`): the success check after each attempt decides whether to retry |
| `SPARK_PHRASES_V2` | off | `1` (token `_ph`): the audited detection phrase table |
| `SPARK_PICK_REGROUND` | `0` | `1` (token `_rg`): the wrist camera re-grounds the pick target before the grasp |
| `SPARK_PICK_REGROUND_GATE_M` | `0.18` | largest accepted re-ground shift in metres (token `_rgs` sets 0.05) |
| `SPARK_FAST_CLOSE` | `1` (`env.sh`: `0`) | `1` ends a gripper close once the fingers settle; `0` runs the full 300 steps |
| `SPARK_IGNORE_DONE` | off (`env.sh`: `1`) | `1` keeps LIBERO's `done` out of the motion loops, so a release finishes |
| `SPARK_OBB_TILT_FIT` | off (`spark_real` sets `1` on import) | `1` grows the top surface of a mask along a tilted table before the box fit |
| `SPARK_OBB_TOP_BAND_M` | `0.008` | height band below the top of a mask that counts as its top surface |
| `SPARK_PROMPT` | unset | `v2` (token `_p2`): the consolidated planner prompt |
| `SPARK_FRESHNESS_STEPS` | `7` | env steps that stand for 50 ms under the `_fs` token |
| `SPARK_DISPLACE_MAGS` | `0,2,5,10` | displacement magnitudes in cm of a `displace` item |
| `SPARK_PLAN_REPLAY` | unset | root of recorded `displace` results whose plans are replayed with no planner call |
| `SPARK_TRACE` | `0` | `1` saves a per-step trace of each trial in `<item>/trace/` |
| `SPARK_TRACE_FRAMES` | `0` | with `SPARK_TRACE=1`, also saves the rendered images every N env steps |

Real robot (`spark_real`); the planner and output variables are in
[src/spark_real/README.md](../src/spark_real/README.md):

| Variable | Default | Effect |
|---|---|---|
| `SPARK_VERIFY_LLM` | on | `0` turns off the planner's image check of a run that geometry could not verify |
| `SPARK_LLM_LAST_RESORT` | `1` | `0` turns off the adjudicate-or-replan rung after a failed check on a UI run |
| `SPARK_START_GATE` | `0` | `1` makes a planned UI run wait for the GO button before its first motion |
| `SPARK_RUN_LOG` | `approved_runs.jsonl` | file under the output root that gets one line per UI run |
| `SPARK_RUN_TAG` | unset | copied into the `tag` field of that line |
| `SPARK_ENABLE_EPISODE_RECORDER` | `0` | `1` records an episode bundle for each UI run |
| `SPARK_VELOCITY_OVERRIDE` | unset | Cartesian speed in m/s for this server process |
| `SPARK_ALLOW_GUESSED_CAMERA_ROLES` | `0` | `1` lets two Kinects without assigned serials take guessed roles instead of raising |
| `SPARK_PRIMARY_CAM` | unset | camera whose detections win the merge (default order: birdview, sideview) |
| `SPARK_RIM_CENTER` | `1` | `0` keeps the mask median as the centre of a deep container |
| `SPARK_GRASP_DEPTH_M` | `0.022` | how far below the perceived top the grasp aims |
| `SPARK_GRASP_CONTACT_N` | `8.0` | force that stops the grasp descent early |
| `SPARK_GRASP_BACKOFF_M` | `0.003` | rise after such a contact, before the close |
| `SPARK_GRASP_SKIP_LIFT_VERIFY` | `1` | `0` forces the lift-based force check even when the gripper's object flag confirms the hold |
| `SPARK_GRASP_VERIFY_SETTLE_S` | `0.4` | wait before the gripper state is read after a close |
| `SPARK_GRASP_CLOSE_FORCE` | `60` | gripper force (0-100) of the re-squeeze when the grip is unconfirmed in transport |
| `SPARK_HELD_DROP_MIN_M` | `0` | floor on the measured hang of the held object below the TCP, capped at 0.15 |
| `SPARK_RELEASE_AT_RIM` | `1` | `0` lowers the object to 2 cm above a deep container's floor instead of releasing above the rim |
| `SPARK_RELEASE_GIVEBACK` | config `place.release_giveback` | overrides it: release height from the measured grasp height |
| `SPARK_RELEASE_OPEN_SETTLE_S` | `0.5` | wait after the jaws are commanded open |
| `SPARK_PLACE_SLOW_FROM_M` | `0` | start the slow place descent at least this far above the release height, capped at 0.12 |
| `SPARK_PLACE_DESCENT_SCALE` | `0.25` | speed of the final place descent as a fraction of the configured velocity |
| `SPARK_STACK_SECOND_LOOK` | `1` | `0` skips the re-detection of the base after the transport of a stack |
| `SPARK_MAX_YAW_OFFSET_DEG` | config `grasp.max_yaw_offset_deg` | largest wrist yaw away from the base orientation |

## Pins

| Component | Pin |
|---|---|
| LIBERO (the `libero` package imported) | Lifelong-Robot-Learning/LIBERO @ 8f1084e3 |
| LIBERO-PRO (scenes, assets, object classes) | Zxy-MLlab/LIBERO-PRO @ eafdb809 |
| LIBERO-PRO perturbed suites | HF dataset zhouxueyang/LIBERO-Pro @ 3a33dc4c |
| SAM3 code | facebookresearch/sam3 @ 5eb25fb5 (the paper runs) + `scripts/sam3.patch` |
| SAM3 checkpoint | facebook/sam3 @ 3c879f39 |
| Gripper close | full 300 steps, as in the paper runs (`SPARK_FAST_CLOSE=0` in env.sh; 1 ends once the fingers settle) |
| Planner weights | MIRALABS/Qwen3.8-27B-W4A16-AutoRound @ a6754efc |
| Planner server | vllm 0.30.0, transformers 5.17.0 |
| Python packages | `uv.lock` |
| Simulator | python 3.12, torch 2.9.1 (cu126), mujoco 3.4.0, robosuite 1.4.1, numpy 2.3.5 |

`sam3.patch` replaces `pkg_resources`, which setuptools 81 and later removed.
LIBERO-PRO must never be importable as `libero`: the protocol runs the upstream package with the
LIBERO-PRO data, and the runner stops if `libero` resolves into LIBERO-PRO.

## Troubleshooting

- `EGL` or `GLContext` errors: check the apt packages and the NVIDIA EGL vendor file above.
- `No module named 'libero'`, or the run waits for input at import: source `scripts/env.sh`.
- `No module named pkg_resources` from sam3: the patch was not applied; re-run `setup.sh`.
- vLLM fails with a `Stale file handle` error: `serve_planner.sh` keeps compile caches in `$TMPDIR`;
  set `TMPDIR` to a local disk.
- On a cluster node without a Hugging Face token, point `HF_HOME` at a cache that already holds
  the pinned SAM3 and Qwen snapshots and set `HF_HUB_OFFLINE=1`.
- Every planner call fails with 404: `SPARK_LLM_MODEL` differs from the served model name.
- Out of GPU memory in a trial: lower `WORKERS` or `VLLM_MEM`. `run_item.sh` treats such a trial
  as failed and runs it again on the next pass.

## Real world deployment

See [real_world.md](real_world.md).
