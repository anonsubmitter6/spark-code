"""scripts/run_item.sh builds the runner command of each repro item (python is stubbed on PATH)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

FAKE_PY = f"""#!/bin/bash
[ "$1" = - ] && exec {sys.executable} "$@"
echo "NO_ORACLE=${{SPARK_NO_ORACLE-unset}} $* OR=${{SPARK_ORACLE_RETRY-}} PH=${{SPARK_PHRASES_V2-}} RG=${{SPARK_PICK_REGROUND-}} RGM=${{SPARK_PICK_REGROUND_GATE_M-}} SK=${{SPARK_SKILLSET-}} MF=${{SPARK_MOTION_FIX-}} FR=${{SPARK_FOLLOW_RETARGET-}} IP=${{SPARK_PICK_INSTANCE_CHECK-}} P2=${{SPARK_PROMPT-}}" >> "$CALLS"
args=("$@")
for i in "${{!args[@]}}"; do
  case "${{args[$i]}}" in
    --dyn-output) echo '{{}}' > "${{args[$((i+1))]}}";;
    --num-trials) n=${{args[$((i+1))]}};;
    --output-dir) out=${{args[$((i+1))]}};;
  esac
done
# a valid result: one task with the asked number of trials (run_item.sh rejects anything else)
[ -n "${{out:-}}" ] && {sys.executable} -c "import json, sys; n = int(sys.argv[2]); json.dump({{'perturbations': {{'p': {{'task_details': [{{'trials': [False] * n, 'trial_meta': [{{}}] * n}}]}}}}}}, open(sys.argv[1] + '/object.json', 'w'))" "$out" "$n"
"""


def _run_item(tmp_path, *item, **leak):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / 'python').write_text(FAKE_PY)
    (bin_dir / 'python').chmod(0o755)
    calls = tmp_path / 'calls.txt'
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", CALLS=str(calls))
    for k in ('SPARK_NO_ORACLE', 'SPARK_ORACLE_RETRY', 'SPARK_PHRASES_V2', 'SPARK_PICK_REGROUND',
              'SPARK_PICK_REGROUND_GATE_M'):
        env.pop(k, None)
    env.update(leak)
    subprocess.run(['bash', str(ROOT / 'scripts' / 'run_item.sh'), str(tmp_path / 'out'), *item],
                   env=env, check=True, capture_output=True, text=True)
    return calls.read_text().split()


def test_displace_arms_match_the_paper_runs(tmp_path):
    on = _run_item(tmp_path, 'displace', 'object', '2', 'on')
    assert on[0] == 'NO_ORACLE=0'
    assert {'--scene-diff', '--layered-recovery', '--no-gemini'} <= set(on)
    assert '--event-captures' not in on
    assert on[on.index('--slide-steps') + 1] == '40'
    assert on[on.index('--phases') + 1] == 'post_plan,mid_approach,post_grasp_plan'
    off = _run_item(tmp_path, 'displace', 'object', '2', 'off')[len(on):]
    assert off[0] == 'NO_ORACLE=0'
    assert not {'--scene-diff', '--layered-recovery', '--event-captures'} & set(off)


def test_headline_item_runs_one_trial_without_oracle_override(tmp_path):
    call = _run_item(tmp_path, 'object', 'position', '3', '7', '1', 'wvrb')
    assert call[0] == 'NO_ORACLE=unset'   # fair/__init__ then defaults it to 1
    assert {'--no-bddl-hints', '--adaptive-prompts', '--wrist-verify', '--relation-bind'} <= set(call)
    assert call[call.index('--only-task-id') + 1] == '3'
    assert call[call.index('--trial-offset') + 1] == '7'
    assert call[call.index('--num-trials') + 1] == '1'


def test_paper_arms_set_their_env(tmp_path):
    t1 = _run_item(tmp_path, 'object', 'position', '0', '0', '1', 'wvrb_or_ph_rgs')
    assert {'--wrist-verify', '--relation-bind', 'OR=1', 'PH=1', 'RG=1', 'RGM=0.05'} <= set(t1)
    strict = _run_item(tmp_path, 'object', 'position', '1', '0', '1', 'wvrb_ph_rgs',
                       SPARK_ORACLE_RETRY='1')[len(t1):]   # a shell export does not leak into the arm
    assert {'OR=', 'PH=1', 'RG=1', 'RGM=0.05'} <= set(strict)
    plain = _run_item(tmp_path, 'object', 'position', '2', '0', '1', 'wvrb')[len(t1) + len(strict):]
    assert {'OR=', 'PH=', 'RG=', 'RGM='} <= set(plain)


def test_strict_displace_arms(tmp_path):
    slide = _run_item(tmp_path, 'displace', 'object', '0', 'onec_st')
    assert slide[0] == 'NO_ORACLE=1'
    assert {'--scene-diff', '--layered-recovery', '--event-captures'} <= set(slide)
    assert slide[slide.index('--slide-steps') + 1] == '40'
    inst = _run_item(tmp_path, 'displace', 'object', '0', 'off_st_inst')[len(slide):]
    assert inst[0] == 'NO_ORACLE=1'
    assert not {'--scene-diff', '--layered-recovery', '--event-captures'} & set(inst)
    assert inst[inst.index('--slide-steps') + 1] == '0'


def test_single_mechanism_displace_arms(tmp_path):
    sd = _run_item(tmp_path, 'displace', 'object', '0', 'sd_st')
    assert '--scene-diff' in sd and '--layered-recovery' not in sd and 'NO_ORACLE=1' in sd
    lr = _run_item(tmp_path, 'displace', 'object', '0', 'lr')[len(sd):]
    assert '--layered-recovery' in lr and '--scene-diff' not in lr and 'NO_ORACLE=0' in lr


def test_llm_displace_arms_vary_only_the_named_mechanism(tmp_path):
    off = _run_item(tmp_path, 'displace', 'object', '0', 'llm-off_st')
    assert '--no-gemini' not in off and '--no-blind-replan' in off and '--scene-diff' not in off
    replan = _run_item(tmp_path, 'displace', 'object', '0', 'llm-replan_st')[len(off):]
    assert '--no-gemini' not in replan and '--no-blind-replan' not in replan
    assert '--layered-recovery' not in replan and 'NO_ORACLE=1' in replan
    on = _run_item(tmp_path, 'displace', 'object', '0', 'llm-on_st')[len(off) + len(replan):]
    assert '--scene-diff' in on and '--layered-recovery' in on


def test_skillset_arm_tokens(tmp_path):
    calls = _run_item(tmp_path, 'object', 'position', '0', '0', '1', 'wvrb_or_ph_rgs_skbase', SPARK_SKILLSET='leak')
    assert 'SK=base' in calls and 'OR=1' in calls and 'RGM=0.05' in calls
    for token, value in (('skshape', 'shape'), ('skask', 'ask'), ('skaskw', 'askw')):
        assert f'SK={value}' in _run_item(tmp_path, 'object', 'position', '0', '0', '1', f'wvrb_or_ph_rgs_pc_{token}')
    calls = _run_item(tmp_path, 'object', 'position', '0', '5', '1', 'wvrb_or_ph_rgs', SPARK_SKILLSET='leak')
    assert 'SK=' in calls


def test_motion_fix_arm_token(tmp_path):
    assert 'MF=1' in _run_item(tmp_path, 'object', 'position', '0', '0', '1', 'wvrb_or_ph_rgs_pc_mf')
    assert 'MF=' in _run_item(tmp_path, 'object', 'position', '0', '5', '1', 'wvrb_or_ph_rgs_pc', SPARK_MOTION_FIX='1')


def test_pick_instance_check_arm_token(tmp_path):
    def run(name, *item, **leak):
        (tmp_path / name).mkdir()
        return _run_item(tmp_path / name, *item, **leak)
    plain = run('a', 'object', 'position', '0', '0', '1', 'wvrb_or_ph_rgs_pc_mf')
    assert 'MF=1' in plain and 'IP=' in plain
    assert 'IP=1' in run('b', 'object', 'position', '0', '0', '1', 'wvrb_or_ph_rgs_pc_mf_ip')
    assert 'IP=1' in run('c', 'displace', 'object', '1', 'llm-lr_st_fs_ud_vs_pc_mf_ip', SPARK_PICK_INSTANCE_CHECK='leak')
    assert 'IP=' in run('d', 'displace', 'object', '1', 'llm-lr_st_fs_ud_vs_pc_mf', SPARK_PICK_INSTANCE_CHECK='1')


def test_prompt_v2_arm_token(tmp_path):
    calls = _run_item(tmp_path, 'object', 'position', '0', '0', '1', 'wvrb_or_ph_rgs_pc_p2_skbase')
    assert 'P2=v2' in calls and 'SK=base' in calls and 'OR=1' in calls
    calls = _run_item(tmp_path, 'object', 'position', '0', '5', '1', 'wvrb_or_ph_rgs_pc', SPARK_PROMPT='v2')
    assert 'P2=' in calls


def test_recovery_option_tokens(tmp_path):
    args = _run_item(tmp_path, 'displace', 'object', '1', 'lr_rf_ud_vs_fs_st')
    for flag in ('--rebind-fresh', '--scene-diff-unknown', '--validator-object-snap', '--layered-recovery'):
        assert flag in args
    assert args[args.index('--freshness-steps') + 1] == '7'


def test_displace_trial_exception_invalidates_the_item(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    fake = FAKE_PY.replace("echo '{}' > ", "echo '{\"cells\": [{\"trials\": [{\"error\": \"CUDA error\"}]}]}' > ")
    assert fake != FAKE_PY
    (bin_dir / 'python').write_text(fake)
    (bin_dir / 'python').chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", CALLS=str(tmp_path / 'calls.txt'))
    subprocess.run(['bash', str(ROOT / 'scripts' / 'run_item.sh'), str(tmp_path / 'out'), 'displace', 'object', '0', 'on'],
                   env=env, capture_output=True, text=True)
    assert list((tmp_path / 'out' / 'failed').iterdir())


def test_libero_pro_recovery_and_dev_tokens(tmp_path):
    a = _run_item(tmp_path, 'object', 'position', '0', '1000', '1', 'wvrb_or_ph_rgs_lr_pc_dev')
    assert {'--layered-recovery', '--post-conditions', '--dev-placements'} <= set(a)
    assert '--no-blind-replan' not in a
    b = _run_item(tmp_path, 'object', 'position', '1', '1000', '1', 'wvrb_ph_rgs_norec')[len(a):]
    assert '--no-blind-replan' in b and '--layered-recovery' not in b and '--dev-placements' not in b


def test_displace_motion_fix_and_added_scene_diff_tokens(tmp_path):
    calls = _run_item(tmp_path, 'displace', 'object', '1', 'llm-replan_sdx_st_fs_ud_vs_pc_mf', SPARK_MOTION_FIX='leak')
    assert 'MF=1' in calls and '--scene-diff' in calls and '--no-blind-replan' not in calls and '--no-gemini' not in calls
    (tmp_path / 'b').mkdir()
    calls = _run_item(tmp_path / 'b', 'displace', 'object', '2', 'llm-replan_st_fs_ud_vs_pc', SPARK_MOTION_FIX='1')
    assert 'MF=' in calls and '--scene-diff' not in calls


def test_follow_retarget_arm_token(tmp_path):
    assert 'FR=1' in _run_item(tmp_path, 'displace', 'object', '1', 'sd_st_fr', SPARK_FOLLOW_RETARGET='leak')
    (tmp_path / 'b').mkdir()
    assert 'FR=' in _run_item(tmp_path / 'b', 'displace', 'object', '2', 'sd_st', SPARK_FOLLOW_RETARGET='1')


def test_displace_seed_and_magnitude_tokens(tmp_path):
    base = _run_item(tmp_path, 'displace', 'object', '0', 'llm-lr_st')
    assert base[base.index('--trial-offset') + 1] == '0' and base[base.index('--magnitudes-cm') + 1] == '0,2,5,10'
    ext = _run_item(tmp_path, 'displace', 'object', '0', 'llm-replan_sdx_st_fs_ud_vs_pc_t5_m510_mf')[len(base):]
    assert ext[ext.index('--trial-offset') + 1] == '5' and ext[ext.index('--magnitudes-cm') + 1] == '5,10'
    assert '--scene-diff' in ext and '--layered-recovery' not in ext and 'NO_ORACLE=1' in ext and 'MF=1' in ext


def test_perception_noise_tokens(tmp_path):
    a = _run_item(tmp_path, 'object', 'task', '0', '1000', '1', 'wvrb_or_ph_rgs_pc_dev_dn2_dd10_seg4_cal0p5_nowrist')
    for flag, v in (('--depth-noise-k', '0.02'), ('--depth-dropout', '0.10'), ('--seg-noise-px', '4'),
                    ('--calib-noise-cm', '0.5'), ('--views', 'agentview')):
        assert a[a.index(flag) + 1] == v
    assert {'--dev-placements', '--post-conditions', 'OR=1', 'RGM=0.05'} <= set(a)
    b = _run_item(tmp_path, 'object', 'task', '0', '1000', '1', 'wvrb_or_ph_rgs_pc_dev_cal1_noagent')[len(a):]
    assert b[b.index('--calib-noise-cm') + 1] == '1' and b[b.index('--views') + 1] == 'wrist'
    c = _run_item(tmp_path, 'object', 'task', '0', '1000', '1', 'wvrb_or_ph_rgs_pc_dev')[len(a) + len(b):]
    assert not {'--depth-noise-k', '--depth-dropout', '--seg-noise-px', '--calib-noise-cm', '--views'} & set(c)
