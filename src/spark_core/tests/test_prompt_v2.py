"""SPARK_PROMPT=v2: the consolidated Franka prompt. Off reads the frozen text; on, each skill-set arm keeps
only the passages whose skills it allows, and the marker lines never reach the model."""
import hashlib
import re
from pathlib import Path

import pytest
import yaml

from spark_core import bt_grammar, skillset
from spark_core.plan_schema import normalize_plan_for_hash
from spark_core.planner import SPARKPlanner
from spark_core.tests.test_prompt_frozen import FAMILY_SHA256

PROMPTS = Path(__file__).resolve().parents[1] / 'prompts'
ALL_SKILLS = set((PROMPTS / 'skill_names.txt').read_text().split())
V2 = (PROMPTS / 'system_franka_v2.txt').read_text()
FULL = skillset.GENERIC | {'open_drawer', 'pull', 'push_object', 'screw', 'search_keypoint', 'tilt_wrist'}


def _planner():
    planner = SPARKPlanner.__new__(SPARKPlanner)
    planner.robot_family = 'franka'
    return planner


def test_switch_off_builds_the_frozen_prompt(monkeypatch):
    monkeypatch.delenv('SPARK_PROMPT', raising=False)
    monkeypatch.delenv('SPARK_SKILLSET', raising=False)
    built = _planner()._build_system_prompt()
    assert hashlib.sha256(built.encode()).hexdigest() == FAMILY_SHA256['system_franka.txt']
    monkeypatch.setenv('SPARK_PROMPT', 'v2')
    assert _planner()._build_system_prompt() == skillset.restrict_prompt(V2) != built


def test_markers_name_real_skills_and_close():
    lines = V2.split('\n')
    names = [n for s in lines if s.startswith('@@needs') for n in s.split()[1:]]
    assert names and set(names) <= ALL_SKILLS
    assert [s for s in lines if s.startswith('@@')] == [s for s in lines if s.startswith('@@needs ') or s == '@@end']
    assert sum(s.startswith('@@needs') for s in lines) == lines.count('@@end')


@pytest.mark.parametrize('arm,allowed', [('', FULL), ('generic', skillset.GENERIC), ('base', skillset.BASE),
                                         ('plain', skillset.BASE), ('shape', skillset.BASE), ('ask', skillset.BASE)])
def test_v2_arm_offers_only_its_skills(monkeypatch, arm, allowed):
    monkeypatch.setenv('SPARK_SKILLSET', arm)
    out = skillset.restrict_prompt(V2)
    assert '@@' not in out and len(out) < 61000 / 3
    head, tail = out.split('Output format (YAML):', 1)
    assert set(re.findall(r'^- (\w+):', head, flags=re.M)) == allowed
    body = out.split('\nSKILL SET:')[0]   # the appended notes name "stack" and "sweep" as words
    # a withheld skill is not named: as a word when its name has an underscore, else as a node, call or `name`
    assert not [n for n in ALL_SKILLS - allowed
                if re.search(rf'\b{n}\b' if '_' in n else rf'type: {n}\b|\b{n}\(|`{n}`', body)]
    assert 'drawer' not in body or 'open_drawer' in allowed
    assert ('SKILL SET:' in out) == bool(arm) and ('BASE-PRIMITIVE' in out) == (arm not in ('', 'generic'))
    assert ('grasp_style' in out) == (arm == 'ask')
    for block in re.findall(r'```yaml\n(.*?)```', out, flags=re.S):   # the examples stay valid YAML in every arm
        yaml.safe_load(block.replace('    ...\n', ''))


def test_v2_advertises_the_enforced_recovery_caps():
    assert f'1 to {bt_grammar.MAX_RETRY_ATTEMPTS}.' in V2
    assert f'At most {bt_grammar.MAX_SELECTOR_BRANCHES}\n  branches' in V2
    assert f'gets {bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE} re-attempts' in V2


def test_assign_key_parses_validates_and_keeps_the_plan_hash():
    plan = ('task: sort\nassign:\n  apple: box 1\n  wrench: box 2\ntree:\n  type: sequence\n  children:\n'
            '    - type: move_to_keypoint\n      params: {keypoint_label: apple, offset_z: 0}\n    - type: grasp\n'
            '    - type: move_to_keypoint\n      params: {keypoint_label: box 1, offset_z: 0.08}\n    - type: release\n')
    planner = _planner()
    score = planner.parse_response(f'```yaml\n{plan}```')
    assert score['assign'] == {'apple': 'box 1', 'wrench': 'box 2'}
    assert planner.validate_score(score, ['apple', 'wrench', 'box 1', 'box 2']) == []
    assert normalize_plan_for_hash(score) == normalize_plan_for_hash({k: v for k, v in score.items() if k != 'assign'})
