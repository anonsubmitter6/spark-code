# Reproduce the paper

All runs need the planner server up (`bash scripts/serve_planner.sh`), except the LIBERO-Displace
arms with scripted plans. Every script resumes: run the same command again to finish or retry
failed items.

## LIBERO-PRO

Mean success (%) over 10 tasks x 50 trials per cell.

| Run | Object pos / task | Goal pos / task | Spatial pos / task | Mean | Command |
|---|---|---|---|---|---|
| Table 1, RATS protocol | 87.8 / 96.0 | 68.6 / 52.2 | 78.0 / 84.2 | 77.8 | `bash scripts/repro_libero_pro.sh table1` |
| Table 1, strict | 84.0 / 94.6 | 62.2 / 44.2 | 78.8 / 82.8 | 74.4 | `bash scripts/repro_libero_pro.sh strict` |

| Run | Object lang / obj | Goal lang / obj | Spatial lang / obj | Mean | Command |
|---|---|---|---|---|---|
| Held out | 92.4 / 77.4 | 64.6 / 47.2 | 50.8 / 85.8 | 69.7 | `bash scripts/repro_libero_pro.sh heldout` |

Summaries:

```bash
python -m spark_bench.summarize results/libero_pro/table1 results/libero_pro/heldout --cells --ci --suite-means
python -m spark_bench.summarize results/libero_pro/strict --timing   # time per trial
```

Each run is one item list in `repro/` (600 lines, 5 trials per line):

    suite perturbation task_id trial_offset n_trials arm

## LIBERO-Displace

Successes out of 45 at 0 / 2 / 5 / 10 cm, verification and recovery on vs off.

| Run | On | Off | Command (`on` arm, then `off` arm) |
|---|---|---|---|
| 40-step slide, success check after each attempt | 39 / 37 / 35 / 30 | 33 / 26 / 7 / 1 | `bash scripts/repro_dyn.sh displace on results/dyn_slide`, `... displace off results/dyn_slide` |
| 40-step slide, strict | 36 / 28 / 15 / 12 | 33 / 25 / 7 / 1 | `... displace onec_st results/dyn_slide_strict`, `... displace off_st ...` |
| Instantaneous, strict | 33 / 25 / 12 / 5 | 33 / 26 / 2 / 0 | `... displace onec_st_inst results/dyn_inst_strict`, `... displace off_st_inst ...` |

Print the table from a run with `python -m spark_bench.dyn.metrics <out_root>`.

### Recovery mechanisms one at a time (ten tasks)

Successes out of 150 at 0 / 2 / 5 / 10 cm: ten object-suite tasks, three displacement phases, five initial
states, strict. Each arm runs with

    TASKS="0 1 2 3 4 5 6 7 8 9" bash scripts/repro_dyn.sh displace <arm> results/disp10/<arm>

The `llm-` arms need the planner server.

| Recovery | Scripted plans | Arm | Planner-written plans | Arm |
|---|---|---|---|---|
| Off | 117 / 87 / 14 / 3 | `off_st_fs_ud_vs_pc` | 120 / 104 / 72 / 71 | `llm-off_st_fs_ud_vs_pc` |
| Scene diff | 105 / 82 / 43 / 8 | `sd_st_fs_ud_vs_pc_fr` | 123 / 101 / 80 / 73 | `llm-sd_st_fs_ud_vs_pc_fr` |
| Ladder | 120 / 108 / 79 / 88 | `lr_st_fs_ud_vs_pc` | 120 / 109 / 85 / 90 | `llm-lr_st_fs_ud_vs_pc` |
| Scene diff + ladder | 123 / 101 / 84 / 90 | `on_st_fs_ud_vs_pc_fr` | 123 / 109 / 99 / 107 | `llm-on_st_fs_ud_vs_pc_fr` |

### Ladder against a new plan after each failure

Planner-written plans on the same displaced trials, same command form.

| Recovery | No scene diff | Arm | With scene diff | Arm |
|---|---|---|---|---|
| New plan | 120 / 104 / 70 / 72 | `llm-replan_st_fs_ud_vs_pc_mf_ip` | 123 / 108 / 84 / 89 | `llm-replan_sdx_st_fs_ud_vs_pc_mf_ip` |
| Ladder | 120 / 111 / 85 / 91 | `llm-lr_st_fs_ud_vs_pc_mf_ip` | 123 / 114 / 95 / 105 | `llm-on_st_fs_ud_vs_pc_mf_ip` |

## Arms

`scripts/run_item.sh` maps the arm token at the end of each item to flags:

| Token | Effect |
|---|---|
| `wv` | wrist camera checks each grasp (`--wrist-verify`) |
| `rb` | spatial relations bind to the right object instance (`--relation-bind`) |
| `_or` | the success check after each attempt decides whether to retry (`SPARK_ORACLE_RETRY=1`) |
| `_ph` | detection phrase table (`SPARK_PHRASES_V2=1`) |
| `_rgs` | wrist camera re-grounds the pick target, at most 5 cm (`SPARK_PICK_REGROUND=1`, `SPARK_PICK_REGROUND_GATE_M=0.05`) |
| `_pc` | grasp and placement checks written in the plan are run (`--post-conditions`) |
| `_mf` | reach check before the close, opposite-rim retry, a failed pick fails its step (`SPARK_MOTION_FIX=1`) |
| `_ip` | with `_mf`: after a failed pick the label stays on its instance only while something still stands there (`SPARK_PICK_INSTANCE_CHECK=1`) |
| `_fr` | the descent follows a retarget made during the move (`SPARK_FOLLOW_RETARGET=1`) |

Without `_or` the run is strict: only the robot's own checks (camera scene diff, empty or slipped
grasp) start a recovery.

## Protocol

- Trial i loads LIBERO-PRO initial state i and never resets after loading, as in LIBERO's own evaluation.
- The planner sees the instruction and the camera images only, never simulator state or the task's
  BDDL file. Every trial plans from scratch; a planner error fails the trial.
- The phrase table and the 5 cm gate were tuned on separate placements (environment seeds 64 to 99),
  never on the evaluation states.
- LIBERO-Displace runs fixed plans, so the arms differ only in verification and recovery. Its runner
  resets after loading to sample new placements.
