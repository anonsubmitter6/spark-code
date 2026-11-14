"""SPARK_SKILLSET: off changes nothing; each arm's prompt and executor offer only that arm's skills."""
import re
from pathlib import Path

import pytest

from spark_core import skillset

PROMPTS = Path(__file__).resolve().parents[1] / 'prompts'
ALL_SKILLS = set((PROMPTS / 'skill_names.txt').read_text().split())


@pytest.mark.parametrize('family', ['ur10e', 'franka'])
def test_off_returns_the_frozen_text(monkeypatch, family):
    monkeypatch.delenv('SPARK_SKILLSET', raising=False)
    text = (PROMPTS / f'system_{family}.txt').read_text()
    assert skillset.restrict_prompt(text) is text
    assert not skillset.is_withheld('open_drawer')
    assert skillset.withheld_leaves({'type': 'open_drawer'}) == []


def test_unknown_arm_raises(monkeypatch):
    monkeypatch.setenv('SPARK_SKILLSET', 'bse')
    with pytest.raises(ValueError):
        skillset.active()


@pytest.mark.parametrize('family', ['ur10e', 'franka'])
@pytest.mark.parametrize('arm,allowed', [('generic', skillset.GENERIC), ('base', skillset.BASE), ('plain', skillset.BASE),
                                          ('shape', skillset.BASE), ('ask', skillset.BASE)])
def test_arm_prompt_offers_only_its_skills(monkeypatch, family, arm, allowed):
    monkeypatch.setenv('SPARK_SKILLSET', arm)
    out = skillset.restrict_prompt((PROMPTS / f'system_{family}.txt').read_text())
    head, tail = out.split('Output format (YAML):', 1)
    assert set(re.findall(r'^- (\w+):', head, flags=re.M)) == allowed
    for name in ALL_SKILLS - allowed:
        # as a node, a call or a backticked name. The words "stack" and "sweep" stay in the base composition text.
        hits = re.findall(rf'type: {name}\b|\b{name}\(|`{name}`', tail)
        assert (not hits) or (name == 'stack' and hits == ['`stack`'] and arm != 'generic'), (name, hits)
    assert 'turn_knob' not in out and 'drawer' not in tail.split('BASE-PRIMITIVE')[0].replace('drawer organizer', '')


def test_withheld_leaves_walks_the_tree(monkeypatch):
    monkeypatch.setenv('SPARK_SKILLSET', 'base')
    tree = {'type': 'sequence', 'children': [
        {'type': 'retry', 'children': [{'type': 'grasp'}, {'type': 'verify_grasp'}]},
        {'type': 'fallback', 'children': [{'type': 'open_drawer'}, {'type': 'release'}]}]}
    assert skillset.withheld_leaves(tree) == ['verify_grasp', 'open_drawer']
    monkeypatch.setenv('SPARK_SKILLSET', 'generic')
    assert skillset.withheld_leaves(tree) == ['open_drawer']
