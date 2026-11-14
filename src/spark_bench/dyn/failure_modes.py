"""Why displacement trials fail: per-trial metrics from traces, a rule-based failure mode, paired tables.

    python -m spark_bench.dyn.failure_modes <out_root> [<out_root> ...] [--json f.json] \
        [--key llm-replan_st llm-lr_st]

Each root holds item directories ``displace_<suite>_<task>_<arm>/`` with ``dyn.json`` and, for traced runs,
``trace/*.npz`` (dyn/trace.py). A trial without a trace gets a coarser mode from its dyn.json records only;
those modes start with ``json:`` so the two kinds never mix in one count.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from spark_bench.dyn.trace import (COL_AIM, COL_AP, COL_ATTEMPT, COL_BODIES, COL_CMD, COL_EE, COL_NOTE, COL_T,
                                   trace_stem)

# The object counts as held when it is this far above its start, this near the gripper, jaws not open.
HOLD_LIFT_M = 0.03
HOLD_EE_M = 0.12
JAWS_OPEN_M = 0.075
# Jaws closed below this at the end of a close met nothing (a held object keeps them apart).
EMPTY_APERTURE_M = 0.005
# The arm "arrived" at a point when slower than this per env step and below this height over the object.
ARRIVED_SPEED_M = 0.002
ARRIVED_HEIGHT_M = 0.15
# An arrival re-aimed when within this of the true object, and nearer it than the pre-displacement spot.
REAIM_GATE_M = 0.04
MIN_SPEED_M = 0.001  # slower than this per env step, a velocity has no meaningful direction
COS_BIN_STEPS = 50
GRASP_NOTES = ('grasp', 'grasp_se3')


def _cos(v: np.ndarray, u: np.ndarray, moving: np.ndarray) -> np.ndarray:
    c = np.full(len(v), np.nan)
    n = np.linalg.norm(v, axis=1) * np.linalg.norm(u, axis=1)
    ok = moving & (n > 0)
    c[ok] = np.sum(v[ok] * u[ok], axis=1) / n[ok]
    return c


def _closes(rows, t, note, types, ee, pick, aim, ap, held) -> list[dict]:
    """One record per grasp note: its first row, and the jaw opening on its last row."""
    closes = []
    for i in np.flatnonzero(np.isin(types, GRASP_NOTES) & (np.diff(note, prepend=-9) != 0)):
        end = i
        while end + 1 < len(note) and note[end + 1] == note[i]:
            end += 1
        closes.append({
            't': int(t[i]), 'attempt': int(rows[i, COL_ATTEMPT]),
            'ee_obj_xy_cm': round(100 * float(np.linalg.norm(ee[i, :2] - pick[i, :2])), 2),
            'ee_obj_z_cm': round(100 * float(ee[i, 2] - pick[i, 2]), 2),
            'ee_obj_xy_end_cm': round(100 * float(np.linalg.norm(ee[end, :2] - pick[end, :2])), 2),
            'aim_obj_xy_cm': round(100 * float(np.linalg.norm(aim[i, :2] - pick[i, :2])), 2),
            'aperture_end_mm': round(1000 * float(ap[end]), 1),
            'lifted_after': bool(held[i:].any()),
        })
    return closes


def trial_metrics(rows: np.ndarray, info: dict, magnitude_cm: float, direction=(0.0, 0.0), slide_steps: int = 0) -> dict:
    """Metrics of one trial; positions come from the simulator, so this is for analysis only."""
    t = rows[:, COL_T]
    ee = rows[:, COL_EE:COL_EE + 3]
    ap = rows[:, COL_AP]
    cmd = rows[:, COL_CMD]
    note = rows[:, COL_NOTE].astype(int)
    aim = rows[:, COL_AIM:COL_AIM + 3]
    bodies = rows[:, COL_BODIES:].reshape(len(rows), -1, 3)
    names = info['body_names']
    initial = np.asarray(info['initial_bodies'], float).reshape(-1, 3)
    pick_i = names.index(info['pick_body'])
    pick, stale = bodies[:, pick_i], initial[pick_i]
    types = np.array([info['notes'][n]['type'] if n >= 0 else 'none' for n in note])
    pr, t_pert = info['perturb_row'], info['perturb_t']
    # Re-aiming is judged from the end of the slide: during it the object is still near its old spot.
    settled = t_pert + slide_steps if t_pert is not None else t[0]
    start = int(np.searchsorted(t, settled)) if pr is not None else 0
    dt = np.diff(t, append=t[-1])

    jaws_on = (cmd >= 0) & (ap < JAWS_OPEN_M)
    held = jaws_on & (pick[:, 2] - stale[2] > HOLD_LIFT_M) & (np.linalg.norm(ee - pick, axis=1) < HOLD_EE_M)
    first_held = int(np.argmax(held)) if held.any() else None

    closes = _closes(rows, t, note, types, ee, pick, aim, ap, held)

    # Did the arm re-aim: an arrival, after the displacement and before any lift, at the true object.
    d_true = np.linalg.norm(ee[:, :2] - pick[:, :2], axis=1)
    d_stale = np.linalg.norm(ee[:, :2] - stale[:2], axis=1)
    stop = first_held if first_held is not None else len(t)
    speed = np.linalg.norm(np.diff(ee[:, :2], axis=0, prepend=ee[:1, :2]), axis=1) / np.maximum(
        np.diff(t, prepend=t[0] - 1), 1)
    arrived = (speed < ARRIVED_SPEED_M) & (ee[:, 2] - pick[:, 2] < ARRIVED_HEIGHT_M)
    window = np.zeros(len(t), bool)
    window[start:stop] = True
    near_true = (d_true < REAIM_GATE_M) & (d_true < d_stale)
    reaim_rows = np.flatnonzero(window & arrived & near_true) if magnitude_cm > 0 else np.array([], int)
    # The same test on where the executor believed the object to be.
    a_true = np.linalg.norm(aim[:, :2] - pick[:, :2], axis=1)
    a_stale = np.linalg.norm(aim[:, :2] - stale[:2], axis=1)
    belief_rows = (np.flatnonzero(window & (a_true < REAIM_GATE_M) & (a_true < a_stale))
                   if magnitude_cm > 0 else np.array([], int))

    # Cosine of EE velocity with the direction to the true object, binned from the displacement on.
    steps = np.maximum(np.diff(t), 1)[:, None]
    v = np.diff(ee, axis=0) / steps
    u = pick[:-1] - ee[:-1]
    moving = np.linalg.norm(v, axis=1) > MIN_SPEED_M
    cos3 = _cos(v, u, moving)
    cos_xy = _cos(v[:, :2], u[:, :2], moving)
    bins: dict[int, list[float]] = defaultdict(list)
    for i in range(start, stop - 1 if stop > start else start):
        if not np.isnan(cos_xy[i]):
            bins[int((t[i] - settled) // COS_BIN_STEPS)].append(float(cos_xy[i]))

    release = np.flatnonzero((types == 'release') & (np.diff(note, prepend=-9) != 0))
    held_at_release = any(held[max(r - 3, 0):r + 1].any() for r in release)
    # How far the object moved before the first close beyond the displacement applied by then.
    knocked = None
    if closes:
        i0 = int(np.flatnonzero(t == closes[0]['t'])[0])
        applied = 0.0 if t_pert is None else float(np.clip((t[i0] - t_pert) / max(slide_steps, 1), 0, 1))
        knocked = round(100 * float(np.linalg.norm(pick[i0, :2] - stale[:2] - applied * np.asarray(direction)[:2])), 2)
    others = []
    for j, name in enumerate(names):
        if j != pick_i and name != info['place_body']:
            b = bodies[:, j]
            if (jaws_on & (b[:, 2] - initial[j, 2] > HOLD_LIFT_M)
                    & (np.linalg.norm(ee - b, axis=1) < HOLD_EE_M)).any():
                others.append(name)
    place_xy = None
    if info['place_body'] in names:
        place_xy = np.linalg.norm(pick[:, :2] - bodies[:, names.index(info['place_body']), :2], axis=1)

    return {
        'n_attempts': int(rows[:, COL_ATTEMPT].max()) + 1,
        'timed_out': info['final_timestep'] >= info['horizon'] > 0,
        'displacement_fired': pr is not None,
        'closes': closes,
        'ever_held': first_held is not None,
        'held_at_release': bool(held_at_release),
        'ever_reaimed': len(reaim_rows) > 0,
        'steps_to_reaim': (int(t[reaim_rows[0]] - t_pert) if len(reaim_rows) and t_pert is not None else None),
        'ever_belief_right': len(belief_rows) > 0,
        'steps_to_belief': (int(t[belief_rows[0]] - t_pert) if len(belief_rows) and t_pert is not None else None),
        'cos_mean': round(float(np.nanmean(cos3)), 3) if np.isfinite(cos3).any() else None,
        'cos_xy_bins': {k: round(float(np.mean(x)), 3) for k, x in sorted(bins.items())},
        'steps_in_note': {k: int(dt[types == k].sum()) for k in sorted(set(types))},
        'knocked_before_close_cm': knocked,
        'other_bodies_lifted': others,
        'final_obj_place_xy_cm': None if place_xy is None else round(100 * float(place_xy[-1]), 1),
        'obj_end_lift_cm': round(100 * float(pick[-1, 2] - stale[2]), 1),
    }


# Ordered: the first rule that holds names the failure, so a trial is named by the furthest stage it reached.
RULES = (
    ('error', lambda m, tm: bool(tm.get('error'))),
    ('timed_out', lambda m, tm: m['timed_out']),
    ('placed_wrong', lambda m, tm: m['held_at_release']),
    ('grasped_dropped', lambda m, tm: m['ever_held']),
    ('wrong_object', lambda m, tm: bool(m['other_bodies_lifted'])),
    ('no_close', lambda m, tm: not m['closes']),
    # Jaws stopped on something, or touched then slid off it (a neck): the gripper's own slip record sees that.
    ('grasped_wrong_place', lambda m, tm: (any(c['aperture_end_mm'] >= 1000 * EMPTY_APERTURE_M for c in m['closes'])
                                           or any(g.get('outcome') == 'slip' for g in tm.get('grasp_outcomes') or []))),
    ('missed_no_displacement', lambda m, tm: not m['displacement_fired']),
    ('never_reaimed', lambda m, tm: not m['ever_reaimed']),
    ('reaimed_missed', lambda m, tm: True),
)


def classify(metrics: Optional[dict], tm: dict) -> str:
    """Failure mode of a trial, or ok_first_try / ok_recovered. Without metrics, the json: modes."""
    if metrics is None:
        return _classify_json(tm)
    if tm.get('success'):
        return 'ok_first_try' if metrics['n_attempts'] <= 1 else 'ok_recovered'
    return next(name for name, rule in RULES if rule(metrics, tm))


def _classify_json(tm: dict) -> str:
    """Coarse mode from the telemetry records alone: what the robot's own gripper reported."""
    g = [x.get('outcome') for x in tm.get('grasp_outcomes') or []]
    recovered = bool(tm.get('recovery_attributions') or tm.get('bt_yaml_recovery') or len(g) > 1)
    if tm.get('success'):
        return 'json:ok_recovered' if recovered else 'json:ok_first_try'
    if tm.get('error'):
        return 'json:error'
    if not g:
        return 'json:no_close'
    if all(x == 'empty_close' for x in g):
        return 'json:only_empty_closes'
    return {'empty_close': 'json:held_earlier_last_empty', 'slip': 'json:last_close_slipped'}.get(
        g[-1], 'json:last_close_secured')


def load_trials(roots) -> dict:
    """{(arm, task, phase, magnitude_cm, trial): record with the trial meta, metrics and mode}."""
    out = {}
    for root in roots:
        for f in sorted(Path(root).glob('displace_*/dyn.json')):
            _, suite, task, arm = f.parent.name.split('_', 3)
            r = json.loads(f.read_text())
            for cell in r['cells']:
                for tm in cell['trials']:
                    p = f.parent / 'trace' / f"{trace_stem(cell['phase'], cell['magnitude_cm'], tm['trial'])}.npz"
                    metrics = None
                    if p.exists():
                        z = np.load(p)
                        rows = z['rows']
                        if len(rows):
                            metrics = trial_metrics(rows, json.loads(str(z['info'])), cell['magnitude_cm'],
                                                    tm.get('direction', (0, 0)), int(tm.get('slide_steps', 0)))
                    key = (arm, f'{suite}_{task}', cell['phase'], cell['magnitude_cm'], tm['trial'])
                    out[key] = {'meta': tm, 'metrics': metrics, 'mode': classify(metrics, tm), 'trace': str(p)}
    return out


def counts_by_arm_magnitude(trials: dict) -> dict:
    tab: dict = defaultdict(Counter)
    for (arm, _task, _phase, mag, _trial), rec in trials.items():
        tab[(arm, mag)][rec['mode']] += 1
    return dict(sorted(tab.items()))


def conversions(trials: dict, arm_a: str, arm_b: str) -> dict:
    """For trials paired by (task, phase, magnitude, trial): {mode in A: [succeed in B, still fail in B]}."""
    out: dict = defaultdict(lambda: [0, 0])
    for (arm, *k), rec in trials.items():
        other = trials.get((arm_b, *k))
        if arm == arm_a and other is not None and not rec['meta'].get('success'):
            out[rec['mode']][0 if other['meta'].get('success') else 1] += 1
    return dict(out)


def _summary(rec: dict) -> dict:
    tm = rec['meta']
    g = [x.get('outcome') for x in tm.get('grasp_outcomes') or []]
    actions = [a.get('action') for a in tm.get('recovery_attributions') or []]
    m = rec['metrics'] or {}
    closes = m.get('closes') or []
    return {'mode': rec['mode'], 'grasp_outcomes': g, 'recovery_actions': actions,
            'n_recovery_plans': len(tm.get('bt_yaml_recovery') or []),
            'n_retargets': len(tm.get('retarget_events') or []),
            'aborts': [a.get('reason') for a in tm.get('scene_aborts') or []],
            'ever_reaimed': m.get('ever_reaimed'), 'steps_to_reaim': m.get('steps_to_reaim'),
            'last_close_err_cm': closes[-1]['ee_obj_xy_cm'] if closes else None}


def paired_table(trials: dict, arm_a: str, arm_b: str) -> dict:
    """Trials that one of two arms solves and the other does not, with how each arm behaved on them."""
    out: dict = {f'{arm_a}_only': [], f'{arm_b}_only': [], 'both': 0, 'neither': 0, 'pairs': 0}
    for (arm, *k), rec in sorted(trials.items(), key=str):
        other = trials.get((arm_b, *k))
        if arm != arm_a or other is None:
            continue
        out['pairs'] += 1
        a, b = bool(rec['meta'].get('success')), bool(other['meta'].get('success'))
        if a == b:
            out['both' if a else 'neither'] += 1
        else:
            out[f'{arm_a if a else arm_b}_only'].append({'trial': k, arm_a: _summary(rec), arm_b: _summary(other)})
    return out


def mode_counts(trials: dict) -> Counter:
    return Counter(rec['mode'] for rec in trials.values())


def _print_paired(tab: dict, arm_a: str, arm_b: str) -> None:
    print(f"\n{arm_a} vs {arm_b}: pairs {tab['pairs']}, both {tab['both']}, neither {tab['neither']}, "
          f"only {arm_a} {len(tab[f'{arm_a}_only'])}, only {arm_b} {len(tab[f'{arm_b}_only'])}")
    for winner, loser in ((arm_a, arm_b), (arm_b, arm_a)):
        rows = tab[f'{winner}_only']
        if not rows:
            continue
        print(f'  {winner} solves, {loser} does not ({len(rows)}):')
        for label, fn in (('task', lambda r: r['trial'][0]), ('phase', lambda r: r['trial'][1]),
                          ('cm', lambda r: r['trial'][2]), (f'{loser} failure', lambda r: r[loser]['mode']),
                          (f'{winner} recovery', lambda r: tuple(r[winner]['recovery_actions'])),
                          (f'{loser} recovery', lambda r: tuple(r[loser]['recovery_actions']))):
            print(f'    by {label}: {dict(Counter(fn(r) for r in rows).most_common())}')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--json', default='', help='write all per-trial metrics and tables here')
    ap.add_argument('--key', nargs=2, action='append', default=[], metavar=('ARM_A', 'ARM_B'),
                    help='print the paired table for two arms; repeatable')
    a = ap.parse_args(argv)
    trials = load_trials(a.roots)
    print(f'{len(trials)} trials, {sum(r["metrics"] is not None for r in trials.values())} with a trace')
    print('Trials per mode:', dict(mode_counts(trials).most_common()))
    print('\nModes per arm and displacement (cm):')
    for (arm, mag), c in counts_by_arm_magnitude(trials).items():
        print(f'  {arm:<18} {mag:>4g}  {dict(c.most_common())}')
    keys = {}
    for arm_a, arm_b in a.key:
        print(f'\nFailures of {arm_a} that {arm_b} converts [succeeds, still fails]:',
              conversions(trials, arm_a, arm_b))
        keys[f'{arm_a}|{arm_b}'] = paired_table(trials, arm_a, arm_b)
        _print_paired(keys[f'{arm_a}|{arm_b}'], arm_a, arm_b)
    if a.json:
        per_trial = {'|'.join(map(str, k)): {x: r[x] for x in ('metrics', 'mode', 'trace')} for k, r in trials.items()}
        Path(a.json).write_text(json.dumps({'trials': per_trial, 'key_tables': keys}, indent=1, default=str))


if __name__ == '__main__':
    main()
