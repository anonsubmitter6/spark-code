"""SPARK_SKILLSET in the LIBERO executor and the planner hint: a withheld leaf is logged, not run."""
from types import SimpleNamespace

from spark_bench.libero_pro import executor as ex
from spark_bench.libero_pro.planning.llm import _hint_instruction


def _exec(calls):
    e = ex.LiberoExecutor.__new__(ex.LiberoExecutor)
    e.abort, e.trial_meta, e.cfg = None, {}, SimpleNamespace(verbose=False)
    e._maybe_refresh_perception = lambda act: None
    e._pre_primitive_gate = lambda i, act: True
    return e


def test_withheld_leaf_is_skipped_and_logged(monkeypatch):
    calls = []
    monkeypatch.setitem(ex._HANDLERS, 'open_drawer', lambda e, p: calls.append('drawer'))
    monkeypatch.setitem(ex._HANDLERS, 'wait', lambda e, p: calls.append('wait'))
    e = _exec(calls)
    monkeypatch.delenv('SPARK_SKILLSET', raising=False)
    assert e.dispatch_action(0, {'type': 'open_drawer'})
    assert calls == ['drawer'] and 'skillset_skipped' not in e.trial_meta
    monkeypatch.setenv('SPARK_SKILLSET', 'base')
    assert e.dispatch_action(1, {'type': 'open_drawer'})
    assert e.dispatch_action(2, {'type': 'wait'})
    assert calls == ['drawer', 'wait'] and e.trial_meta['skillset_skipped'] == ['open_drawer']


def test_push_hint_follows_the_arm(monkeypatch):
    args = ('push the plate', '', '', None, True, None, None, None)
    monkeypatch.delenv('SPARK_SKILLSET', raising=False)
    assert 'push_object' in _hint_instruction(*args)[0]
    monkeypatch.setenv('SPARK_SKILLSET', 'generic')
    assert 'push_object' not in _hint_instruction(*args)[0]


def test_v2_prompt_offers_every_frozen_skill_the_simulator_runs(monkeypatch):
    import re
    from spark_bench.libero_pro.conditions import CONDITIONS
    from spark_core import skillset
    from spark_core.planner import PROMPTS_DIR
    monkeypatch.delenv('SPARK_SKILLSET', raising=False)

    def offered(name):
        head = skillset.restrict_prompt((PROMPTS_DIR / name).read_text()).split('Output format (YAML):')[0]
        return set(re.findall(r'^- (\w+):', head, flags=re.M))

    runs = set(ex._HANDLERS) | set(CONDITIONS)
    assert offered('system_franka_v2.txt') == offered('system_franka.txt') & runs
