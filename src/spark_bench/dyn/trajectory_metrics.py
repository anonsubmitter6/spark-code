"""Trajectory failure tracking for displacement trials: action cosine, phase-aligned DTW, automatic labels.

    python -m spark_bench.dyn.trajectory_metrics --dyn-root D --trace-root T --out O [--workers 6]

Analysis only: nothing here feeds the system and no runner imports it. Traces come from dyn/trace.py. The true
object positions in a trace are read for analysis. Signals named ``gt_*`` use them, signals named ``ng_*`` use
only what the robot has (end effector, jaws, the executor's own believed position in ``COL_AIM``).

Methods follow the DTW and cosine definitions of the author's injection study: DTW is Euclidean local cost
over end-effector positions, normalized by (n + m); action cosine flattens two action sequences and takes one
cosine. Phase alignment splits an attempt into roles by behavior-tree note and runs DTW per role.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from spark_bench.dyn.failure_modes import (HOLD_LIFT_M, JAWS_OPEN_M, MIN_SPEED_M, REAIM_GATE_M, _cos, classify,
                                           trial_metrics)
from spark_bench.dyn.trace import (COL_AIM, COL_AP, COL_ATTEMPT, COL_BODIES, COL_CMD, COL_EE, COL_NOTE, COL_T,
                                   trace_stem)

ROLES = ('approach', 'grasp', 'lift', 'carry', 'release')
FRAMES = ('world', 'obj', 'aim')
RESAMPLE = 20
TRACK_M = 0.06    # a body within this of the gripper while it rises is being carried
LATE_FRACTION = 1 / 3  # "late" cosine: the last third of the approach, where a re-aim shows
MAX_REFS = 8
CLUSTER_CM = 2.0  # average-link cut on mean per-role DTW; a constant offset d gives a DTW of d/2
CLUSTER_SAMPLE = 60
HOLD_END_M = 0.04  # object this near the gripper at the end, jaws not open: never let go
GOAL_XY_M = 0.06   # object this near the basket center in xy at the end


def dtw_distance(a, b) -> float:
    """DTW between [T, d] sequences, Euclidean local cost, D[n, m] / (n + m)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float('inf')
    a, b = np.asarray(a, float).reshape(n, -1), np.asarray(b, float).reshape(m, -1)
    cost = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).tolist()
    prev = [float('inf')] * (m + 1)
    prev[0] = 0.0
    for i in range(n):
        ci = cost[i]
        cur = [float('inf')] * (m + 1)
        left = cur[0]
        for j in range(m):
            best = min(prev[j], prev[j + 1], left)
            left = ci[j] + best
            cur[j + 1] = left
        prev = cur
    return prev[m] / (n + m)


def action_cosine(a, b) -> float:
    """cos(a, b) over the flattened action vectors of two equal-length sequences."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float('nan')


def resample(path: np.ndarray, n: int = RESAMPLE) -> np.ndarray:
    """``n`` points evenly spaced along arc length, so dwell time does not weigh in the DTW."""
    path = np.asarray(path, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if s[-1] <= 0:
        return np.repeat(path[:1], n, axis=0)
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, path[:, d]) for d in range(path.shape[1])], axis=1)


def roles_of(info: dict, attempt: int) -> dict:
    """{role: note indices} of one attempt. Roles come from order and note type, not from labels."""
    out, seen_grasp, seen_kp = defaultdict(list), False, False
    for i, n in enumerate(info['notes']):
        if n['attempt'] != attempt or n['idx'] < 0:
            continue
        t = n['type']
        if t.startswith('verify'):  # post-condition checks hold still; they would pollute a motion role
            continue
        if t.startswith('grasp'):
            role, seen_grasp = 'grasp', True
        elif t == 'release':
            role = 'release'
        elif not seen_grasp:
            role = 'approach'
        elif t == 'move_to_keypoint' or seen_kp:
            role, seen_kp = 'carry', True
        else:
            role = 'lift'
        out[role].append(i)
    return dict(out)


def _seg(note: np.ndarray, notes: list) -> np.ndarray:
    return np.flatnonzero(np.isin(note, notes))


def _mean(x) -> float:
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')


def _cos_seg(ee, u, t, ix) -> np.ndarray:
    """Cosine of EE motion with vector ``u`` on consecutive rows of ``ix``."""
    if len(ix) < 2:
        return np.array([])
    i, j = ix[:-1], ix[1:]
    v = (ee[j] - ee[i]) / np.maximum(t[j] - t[i], 1)[:, None]
    return _cos(v, u[i], np.linalg.norm(v, axis=1) > MIN_SPEED_M)


def attempt_features(rows, info, attempt) -> tuple:
    """(signals, role paths per frame) of one attempt; ``gt_*`` read true positions, ``ng_*`` do not."""
    t, ee, ap = rows[:, COL_T], rows[:, COL_EE:COL_EE + 3], rows[:, COL_AP]
    cmd, note, aim = rows[:, COL_CMD], rows[:, COL_NOTE].astype(int), rows[:, COL_AIM:COL_AIM + 3]
    bodies = rows[:, COL_BODIES:].reshape(len(rows), -1, 3)
    names = info['body_names']
    pick_i, place_i = names.index(info['pick_body']), names.index(info['place_body'])
    pick, place = bodies[:, pick_i], bodies[:, place_i]
    in_att = np.flatnonzero(rows[:, COL_ATTEMPT] == attempt)
    rmap = roles_of(info, attempt)
    seg = {r: _seg(note, rmap[r]) if r in rmap else np.array([], int) for r in ROLES}
    f = {'n_roles': sum(len(s) > 0 for s in seg.values())}
    app, gr, lift, carry, rel = (seg[r] for r in ROLES)

    # Cosines: arm motion with the direction to the true object or goal, and to the believed one.
    f['gt_cos_obj'] = _mean(_cos_seg(ee, pick - ee, t, app))
    f['ng_cos_aim'] = _mean(_cos_seg(ee, aim - ee, t, app))
    if len(app) >= 3:
        late = app[int(len(app) * (1 - LATE_FRACTION)):]
        f['gt_cos_obj_late'] = _mean(_cos_seg(ee, pick - ee, t, late))
        f['ng_cos_aim_late'] = _mean(_cos_seg(ee, aim - ee, t, late))
    else:
        f['gt_cos_obj_late'] = f['ng_cos_aim_late'] = float('nan')
    f['gt_cos_goal'] = _mean(_cos_seg(ee, place - ee, t, carry))
    f['ng_cos_aim_carry'] = _mean(_cos_seg(ee, aim - ee, t, carry))

    # Distances at the close and across the attempt.
    if len(app):
        d = np.linalg.norm(ee[app] - pick[app], axis=1)
        f['gt_closing'] = float(d[0] - d[-1])
        f['gt_app_end'] = float(d[-1])
        da = np.linalg.norm(ee[app] - aim[app], axis=1)
        f['ng_aim_gap'] = float(da[-1]) if np.isfinite(da[-1]) else float('nan')
        f['ng_aim_shift'] = float(np.nan_to_num(np.linalg.norm(aim[app[-1], :2] - aim[app[0], :2])))
    close_xy = float('nan')
    nearest = None
    if len(gr):
        dxy = np.linalg.norm(ee[gr[0], :2] - bodies[gr[0], :, :2], axis=1)
        close_xy = float(dxy[pick_i])
        order = [j for j in np.argsort(dxy) if j != place_i]
        nearest, f['near_xy'] = names[order[0]], float(dxy[order[0]])
        f['ng_ap_close'] = float(ap[gr[-1]])
    f['gt_close_xy'] = close_xy
    f['gt_stale_xy'] = (float(np.linalg.norm(ee[gr[0], :2] - np.asarray(info['initial_bodies'][pick_i][:2], float)))
                        if len(gr) else float('nan'))
    f['nearest_at_close'] = nearest

    # What rose with the gripper during this attempt.
    jaws_on = (cmd >= 0) & (ap < JAWS_OPEN_M)
    start_z = bodies[in_att[0], :, 2] if len(in_att) else bodies[0, :, 2]
    lifted = []
    if len(in_att):
        for j, name in enumerate(names):
            if j == place_i:
                continue
            carried = (jaws_on[in_att] & (bodies[in_att, j, 2] - start_z[j] > HOLD_LIFT_M)
                       & (np.linalg.norm(ee[in_att] - bodies[in_att, j], axis=1) < TRACK_M))
            if carried.any():
                lifted.append(name)
    f['lifted'] = lifted
    f['gt_pick_lifted'] = info['pick_body'] in lifted
    f['gt_track_carry'] = float(np.linalg.norm(ee[carry[-1]] - pick[carry[-1]])) if len(carry) else float('nan')

    # End of the attempt, settling rows included.
    if len(in_att):
        e = in_att[-1]
        f['gt_goal_xy'] = float(np.linalg.norm(pick[e, :2] - place[e, :2]))
        f['gt_goal_dz'] = float(pick[e, 2] - place[e, 2])
        f['gt_held_end'] = bool(np.linalg.norm(ee[e] - pick[e]) < HOLD_END_M and ap[e] < JAWS_OPEN_M)
        f['ng_ap_end'] = float(ap[e])
        f['ng_jaws_closed_end'] = bool(ap[e] < JAWS_OPEN_M)
        f['ng_release_steps'] = float(t[rel[-1]] - t[rel[0]]) if len(rel) else 0.0
        # The belief about the pick when the attempt ends, against the place the arm carried to.
        goal_aim = aim[carry[-1], :2] if len(carry) else aim[e, :2]
        last_pick_aim = aim[rel[-1], :2] if len(rel) else aim[e, :2]
        f['ng_aim_end_to_place'] = float(np.linalg.norm(last_pick_aim - goal_aim))
        f['t_end'] = float(t[e])

    # Role paths per frame. "obj" subtracts the true pick position, row by row, so a correct re-aim costs nothing.
    paths: dict = {fr: {} for fr in FRAMES}
    for r in ROLES:
        ix = seg[r]
        if len(ix) == 0:
            continue
        paths['world'][r] = resample(ee[ix]).astype(np.float32)
        paths['obj'][r] = resample(ee[ix] - pick[ix]).astype(np.float32)
        a = aim[ix]
        if np.isfinite(a).all():
            paths['aim'][r] = resample(ee[ix] - a).astype(np.float32)
    return f, paths


def path_distance(p, q) -> tuple:
    """Mean per-role DTW over the roles both have, and the per-role values."""
    d = {r: dtw_distance(p[r], q[r]) for r in ROLES if r in p and r in q}
    return (float(np.mean(list(d.values()))) if d else float('inf')), d


def failure_label(tr: dict) -> str:
    """What a trial's trace shows, from signals alone: the simulator's success is not read here.

    ``in_goal`` and ``held_over_goal`` are the two endings the predicate can accept; geometry cannot tell them
    from failure exactly, because the predicate needs contact and a region that depends on the object. The
    others name the furthest stage reached in any attempt.
    """
    atts, last = tr['attempts'], tr['attempts'][-1]
    if tr['timed_out']:
        return 'timed_out'
    if last.get('gt_goal_xy', 9) < GOAL_XY_M:
        return 'held_over_goal' if last.get('gt_held_end') else 'in_goal'
    for a in reversed(atts):
        if a['gt_pick_lifted']:
            return 'placed_wrong' if a['gt_track_carry'] < TRACK_M else 'dropped'
    if any(a['lifted'] for a in atts):
        return 'wrong_object'
    if tr['mag'] == 0:
        return 'missed_no_displacement'
    # Re-aimed: a close at the true object and nearer it than the old spot, or the approach path nearer the
    # shifted same-seed path with its late motion toward the object.
    closed = any(a.get('gt_close_xy', 9) < REAIM_GATE_M and a['gt_close_xy'] < a.get('gt_stale_xy', 9) for a in atts)
    aimed = any(a.get('gt_cos_obj_late', -1) > 0.5 and a.get('stale_dominant') is False for a in atts)
    return 'reaimed_missed' if closed or aimed else 'never_reaimed'


# loading

def _load_item(job) -> list:
    f, trace_root = job
    item = Path(f).parent.name
    m = re.match(r'displace_(\w+?)_(\d+)_(.*)$', item)
    arm, task = re.sub(r'(_fs|_ud|_vs)', '', m.group(3)), int(m.group(2))
    r = json.loads(Path(f).read_text())
    out = []
    for c in r['cells']:
        for tm in c['trials']:
            p = Path(trace_root) / item / 'trace' / f"{trace_stem(c['phase'], c['magnitude_cm'], tm['trial'])}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            rows, info = z['rows'], json.loads(str(z['info']))
            metr = trial_metrics(rows, info, c['magnitude_cm'], tm.get('direction', (0, 0)), int(tm.get('slide_steps', 0)))
            n = int(rows[:, COL_ATTEMPT].max()) + 1
            feats = [attempt_features(rows, info, a) for a in range(n)]
            outcomes = [g.get('outcome') for g in tm.get('grasp_outcomes') or []]
            labels = [l for l in (x['label'] for x in info['notes'] if x['idx'] >= 0) if l]
            seqs = []
            for a in range(n):
                seqs.append(tuple((x['type'], x['label']) for x in info['notes'] if x['attempt'] == a and x['idx'] >= 0))
            # World path of the whole trial, all attempts, for the naive whole-episode score.
            whole = resample(rows[:, COL_EE:COL_EE + 3], 60).astype(np.float32)
            out.append(dict(
                arm=arm, task=task, name=r['task_name'], phase=c['phase'], mag=float(c['magnitude_cm']),
                trial=tm['trial'], success=bool(tm.get('success')), n_attempts=n,
                timed_out=info['final_timestep'] >= info['horizon'] > 0, direction=tm.get('direction', [0, 0]),
                rule=classify(metr, tm), attempts=[x[0] for x in feats], paths=[x[1] for x in feats],
                outcomes=outcomes, seqs=seqs, pick_label=labels[0] if labels else None, whole=whole,
                n_recovery=len(tm.get('recovery_attributions') or []),
                recovery=[(a.get('action'), a.get('diff_status')) for a in tm.get('recovery_attributions') or []],
                n_retarget=len(tm.get('retarget_events') or []), binding=tm.get('final_binding') or {},
                detected_by=tm.get('detected_by'), adapted_by=tm.get('adapted_by')))
    return out


def load_all(dyn_root, trace_root, workers) -> list:
    jobs = [(f, trace_root) for f in sorted(glob.glob(f'{dyn_root}/displace_*/dyn.json'))]
    with Pool(workers) as pool:
        return [t for chunk in pool.imap(_load_item, jobs) for t in chunk]


def _key(t: dict) -> tuple:
    return t['arm'], t['task'], t['phase'], t['mag'], t['trial']


# references: several per task, held out by trial index

def _cluster_job(job) -> tuple:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    task, held, frame, paths = job
    n = len(paths)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = path_distance(paths[i], paths[j])[0]
    D[~np.isfinite(D)] = 1.0
    out = {'task': task, 'held': held, 'frame': frame, 'n': n, 'counts': {}, 'one': None, 'many': []}
    if n == 0:
        return out
    out['one'] = paths[int(np.argmin(D.sum(1)))]
    if n == 1:
        out['many'] = [paths[0]]
        out['counts'] = {1: 1, 2: 1, 4: 1}
        return out
    Z = linkage(squareform(D, checks=False), 'average')
    for cm in (1, 2, 4):
        out['counts'][cm] = int(fcluster(Z, cm / 100, 'distance').max())
    lab = fcluster(Z, CLUSTER_CM / 100, 'distance')
    sizes = Counter(lab.tolist())
    out['sizes'] = sorted(sizes.values(), reverse=True)
    # Leave-one-cluster-out: how far is each path from the medoids of the other clusters.
    meds = {}
    for c in sizes:
        ix = np.flatnonzero(lab == c)
        meds[c] = ix[int(np.argmin(D[np.ix_(ix, ix)].sum(1)))]
    if held == -1:
        loo = []
        for i in range(n):
            others = [D[i, m] for c, m in meds.items() if c != lab[i]]
            own = D[i, meds[lab[i]]]
            if others:
                loo.append((float(own), float(min(others))))
        out['leave_cluster_out'] = loo
    out['many'] = [paths[meds[c]] for c, _ in sizes.most_common(MAX_REFS)]
    return out


def build_references(trials, workers, rng_seed=0) -> dict:
    """{(task, held trial or -1, frame): cluster record}, from the last attempts of successful trials."""
    rng = np.random.default_rng(rng_seed)
    by_task = defaultdict(list)
    for t in trials:
        if t['success']:
            by_task[t['task']].append(t)
    jobs = []
    for task, ts in sorted(by_task.items()):
        for held in (-1, *sorted({t['trial'] for t in trials})):
            pool = [t for t in ts if t['trial'] != held]
            if len(pool) > CLUSTER_SAMPLE:
                pool = [pool[i] for i in rng.choice(len(pool), CLUSTER_SAMPLE, replace=False)]
            for frame in FRAMES:
                jobs.append((task, held, frame, [t['paths'][-1][frame] for t in pool]))
    with Pool(workers) as pool:
        return {(r['task'], r['held'], r['frame']): r for r in pool.imap_unordered(_cluster_job, jobs, chunksize=4)}


def _nearest(paths, refs) -> tuple:
    best, broles, idx = float('inf'), {}, -1
    for i, q in enumerate(refs):
        d, per = path_distance(paths, q)
        if d < best:
            best, broles, idx = d, per, i
    return best, broles, idx


_REFS: dict = {}


def _init_refs(refs) -> None:
    global _REFS
    _REFS = refs


def _score_trial(job) -> tuple:
    """DTW of the first and last attempts to their references, per frame, and whole-trial and stale checks."""
    key, t, whole_refs, base = job
    refs = _REFS
    res = {}
    for tag, a in (('first', 0), ('last', t['n_attempts'] - 1)):
        rec = {}
        pa = t['paths'][a]
        for frame in FRAMES:
            r = refs.get((t['task'], t['trial'], frame))
            if r and r['many']:
                rec[f'{frame}_one'], rec[f'{frame}_one_roles'] = path_distance(pa[frame], r['one'])
                rec[f'{frame}_many'], rec[f'{frame}_many_roles'], i = _nearest(pa[frame], r['many'])
                if frame == 'world' and i >= 0:
                    p, q = pa[frame], r['many'][i]
                    common = [x for x in ROLES if x in p and x in q]
                    if common:
                        rec['act_cos_nn'] = action_cosine(np.concatenate([np.diff(p[x], axis=0) for x in common]),
                                                          np.concatenate([np.diff(q[x], axis=0) for x in common]))
        # Nearest reference from each other task: the fold chooses which tasks count.
        rec['xtask'] = {}
        for frame in ('world', 'obj'):
            rec['xtask'][frame] = {}
            for other in range(10):
                r = refs.get((other, -1, frame))
                if other != t['task'] and r and r['many']:
                    d, per, _ = _nearest(pa[frame], r['many'])
                    d1, per1 = path_distance(pa[frame], r['one'])
                    rec['xtask'][frame][other] = (d, per, d1, per1)
        # Action cosine with the same seed at 0 cm with no recovery, and stale-against-shifted dominance on the approach.
        if base is not None:
            pb = base[0]
            common = [x for x in ROLES if x in pa['world'] and x in pb]
            if common:
                rec['act_cos_base'] = action_cosine(
                    np.concatenate([np.diff(pa['world'][x], axis=0) for x in common]),
                    np.concatenate([np.diff(pb[x], axis=0) for x in common]))
            if 'approach' in pa['world'] and 'approach' in pb and t['mag'] > 0:
                shift = np.r_[t['direction'], 0.0]
                rec['d_stale'] = dtw_distance(pa['world']['approach'], pb['approach'])
                rec['d_true'] = dtw_distance(pa['world']['approach'], pb['approach'] + shift)
        res[tag] = rec
    whole = whole_refs.get((t['task'], t['trial']))
    if whole:
        res['whole'] = min(dtw_distance(t['whole'], q) for q in whole)
    return key, res


def score_all(trials, refs, workers) -> dict:
    by_key = {_key(t): t for t in trials}
    # Whole-trial references: successful first-try whole paths of the same task, other trial index.
    wr = defaultdict(list)
    idxs = sorted({t['trial'] for t in trials})
    rng = np.random.default_rng(1)
    for task in sorted({t['task'] for t in trials}):
        ok = [t for t in trials if t['task'] == task and t['success']]
        for held in idxs:
            pool = [t for t in ok if t['trial'] != held]
            if len(pool) > 8:
                pool = [pool[i] for i in rng.choice(len(pool), 8, replace=False)]
            wr[(task, held)] = [t['whole'] for t in pool]
    jobs = []
    for k, t in by_key.items():
        b = by_key.get(('off', t['task'], t['phase'], 0.0, t['trial']))
        base = (b['paths'][0]['world'],) if b is not None and k != _key(b) else None
        slim = {x: t[x] for x in ('task', 'trial', 'n_attempts', 'mag', 'direction', 'whole')}
        slim['paths'] = t['paths']
        jobs.append((k, slim, {(t['task'], t['trial']): wr[(t['task'], t['trial'])]}, base))
    out = {}
    with Pool(workers, initializer=_init_refs, initargs=(refs,)) as pool:
        for i, (k, r) in enumerate(pool.imap_unordered(_score_trial, jobs, chunksize=16)):
            out[k] = r
            if (i + 1) % 1000 == 0:
                print(f'  scored {i + 1}/{len(jobs)}', flush=True)
    return out


# held-out evaluation

COS = ['gt_cos_obj', 'gt_cos_obj_late', 'gt_cos_goal', 'dcos_obj', 'dcos_goal', 'act_cos_base', 'act_cos_nn']
DIST = ['gt_closing', 'gt_app_end', 'gt_close_xy', 'gt_lift', 'gt_track_carry']
FINAL = ['gt_goal_xy', 'gt_goal_dz', 'gt_held']
NOGT = ['ng_cos_aim', 'ng_cos_aim_late', 'ng_cos_aim_carry', 'ng_aim_gap', 'ng_aim_shift', 'ng_ap_close', 'ng_ap_end',
        'ng_aim_end_to_place', 'ng_ap_dev', 'ng_alias_spread', 'g_secured', 'g_empty', 'g_slip']
# Both are consequences of env.step's done flag cutting the release short, so they carry the predicate in this sim.
DONE_LEAK = ['ng_release_steps', 'ng_jaws_closed']
OPS = (0.01, 0.05, 0.10)


def _dtw_cols(frame, kind) -> list:
    return [f'{frame}_{kind}'] + [f'{frame}_{kind}_{r}' for r in ROLES]


def _xcols(frame, kind) -> list:
    return [f'x{frame}_{kind}'] + [f'x{frame}_{kind}_{r}' for r in ROLES]


DETECTORS = {
    'cosine': COS,
    'dist_no_final': DIST,
    'dtw_world_one': _dtw_cols('world', 'one'),
    'dtw_world_many': _dtw_cols('world', 'many'),
    'dtw_obj_one': _dtw_cols('obj', 'one'),
    'dtw_obj_many': _dtw_cols('obj', 'many'),
    'dtw_world_many_xtask': _xcols('world', 'many'),
    'dtw_obj_one_xtask': _xcols('obj', 'one'),
    'dtw_obj_many_xtask': _xcols('obj', 'many'),
    'cos+dtw_obj_many': COS + _dtw_cols('obj', 'many'),
    'cos+dtw_obj_many_xtask': COS + _xcols('obj', 'many'),
    'cos+dist': COS + DIST,
    'cos+dist+dtw_many': COS + DIST + _dtw_cols('obj', 'many'),
    'final_distance(near_label)': FINAL,
    'nogt_only': NOGT,
    'nogt_no_refs': [c for c in NOGT if c != 'ng_ap_dev'],
    'nogt+dtw_world': NOGT + _dtw_cols('world', 'many') + _dtw_cols('aim', 'many'),
    'nogt+dtw_world+done_leak': NOGT + DONE_LEAK + _dtw_cols('world', 'many') + _dtw_cols('aim', 'many'),
}


def _alias_spread(t) -> float:
    """Largest xy gap between the bound positions of the pick's labels (the place's labels left out)."""
    b = t['binding'] or {}
    place = set()
    for s in t['seqs']:
        seen = False
        for typ, lab in s:
            seen = seen or typ.startswith('grasp')
            if seen and lab:
                place |= set(lab.lower().split())
    pts = [np.asarray(v[:2], float) for k, v in b.items() if v is not None and not set(k.lower().split()) & place]
    if len(pts) < 2:
        return 0.0
    return float(max(np.linalg.norm(a - b_) for i, a in enumerate(pts) for b_ in pts[i + 1:]))


def build_rows(trials, scores, ref_trials=None) -> list:
    """One row per trial and attempt tag, with every signal; the DTW cross-task entries stay per reference task."""
    by_key = {_key(t): t for t in trials}
    ap_refs = defaultdict(list)
    for t in (trials if ref_trials is None else ref_trials):
        if t['success'] and 'ng_ap_close' in t['attempts'][-1]:
            ap_refs[t['task']].append((t['trial'], t['attempts'][-1]['ng_ap_close']))
    rows = []
    for k, t in by_key.items():
        b = by_key.get(('off', t['task'], t['phase'], 0.0, t['trial']))
        for tag, a in (('first', 0), ('last', t['n_attempts'] - 1)):
            f, rec = t['attempts'][a], scores[k][tag]
            r = {x: f.get(x, float('nan')) for x in COS + DIST + FINAL + NOGT + DONE_LEAK if x in f}
            r['gt_lift'] = float(bool(f.get('gt_pick_lifted')))
            r['gt_held'] = float(bool(f.get('gt_held_end')))
            r['ng_jaws_closed'] = float(bool(f.get('ng_jaws_closed_end')))
            r['g_secured'], r['g_empty'], r['g_slip'] = 0.0, 0.0, 0.0
            if t['outcomes']:
                # Outcomes are per close, in order; each attempt has at most one here (a local retry could add more).
                o = t['outcomes'][min(a, len(t['outcomes']) - 1)]
                r['g_secured'], r['g_empty'], r['g_slip'] = float(o == 'secured'), float(o == 'empty_close'), float(o == 'slip')
            pool = [x for tr, x in ap_refs[t['task']] if tr != t['trial']]
            r['ng_ap_dev'] = abs(f.get('ng_ap_close', np.nan) - float(np.median(pool))) if pool else float('nan')
            r['ng_alias_spread'] = _alias_spread(t)
            if b is not None and _key(b) != k:
                bf = b['attempts'][0]
                r['dcos_obj'] = f.get('gt_cos_obj', np.nan) - bf.get('gt_cos_obj', np.nan)
                r['dcos_goal'] = f.get('gt_cos_goal', np.nan) - bf.get('gt_cos_goal', np.nan)
            r['act_cos_base'] = rec.get('act_cos_base', float('nan'))
            r['act_cos_nn'] = rec.get('act_cos_nn', float('nan'))
            for frame in FRAMES:
                for kind in ('one', 'many'):
                    r[f'{frame}_{kind}'] = rec.get(f'{frame}_{kind}', float('nan'))
                    per = rec.get(f'{frame}_{kind}_roles') or {}
                    for role in ROLES:
                        r[f'{frame}_{kind}_{role}'] = per.get(role, float('nan'))
            r['_xtask'] = rec.get('xtask')
            if rec.get('d_stale') is not None:
                r['d_stale'], r['d_true'] = rec['d_stale'], rec['d_true']
            rows.append(dict(key=k, tag=tag, r=r))
    return rows


def _xvalue(r, name, held) -> float:
    """Cross-task DTW column, nearest over reference tasks other than this trial's and the held-out task."""
    x = r['_xtask']
    _, frame_kind = name[0], name[1:]
    parts = frame_kind.split('_')
    frame, kind = parts[0], parts[1]
    role = parts[2] if len(parts) > 2 else None
    best, bval = float('inf'), float('nan')
    for other, (dm, pm, d1, p1) in x[frame].items():
        if other == held:
            continue
        d, per = (dm, pm) if kind == 'many' else (d1, p1)
        if d < best:
            best, bval = d, (d if role is None else per.get(role, float('nan')))
    return bval


def _matrix(rows, cols, held=None) -> np.ndarray:
    M = np.full((len(rows), len(cols)), np.nan)
    for i, row in enumerate(rows):
        r = row['r']
        for j, c in enumerate(cols):
            if c.startswith('xworld') or c.startswith('xobj'):
                M[i, j] = _xvalue(r, c, held if held is not None else -99)
            else:
                M[i, j] = r.get(c, np.nan)
    return M


def _logistic(X, y, l2=1.0, iters=25) -> np.ndarray:
    """Ridge logistic regression by Newton steps, on standardized columns with an intercept."""
    X1 = np.c_[np.ones(len(X)), X]
    w = np.zeros(X1.shape[1])
    R = l2 * np.eye(X1.shape[1])
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ w, -30, 30)))
        g = X1.T @ (p - y) + R @ w
        H = (X1 * (p * (1 - p) + 1e-6)[:, None]).T @ X1 + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-5:
            break
    return w


def _fit_score(Xtr, ytr, Xte) -> tuple:
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr, Xte = np.where(np.isfinite(Xtr), Xtr, med), np.where(np.isfinite(Xte), Xte, med)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = _logistic(Xtr, ytr.astype(float))
    return w[0] + Xtr @ w[1:], w[0] + Xte @ w[1:], w


def auc(score, y) -> float:
    from scipy.stats import rankdata
    ok = np.isfinite(score)
    score, y = score[ok], y[ok].astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((rankdata(score)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _task_ci(score, y, tasks, reps=1000, seed=0) -> tuple:
    """Percentile CI of the pooled AUC, resampling whole tasks (trials within a task are not independent)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(tasks)
    ix = {u: np.flatnonzero(tasks == u) for u in uniq}
    vals = []
    for _ in range(reps):
        take = np.concatenate([ix[u] for u in rng.choice(uniq, len(uniq))])
        vals.append(auc(score[take], y[take]))
    v = np.array(vals)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (float('nan'), float('nan'))


def loto(rows, y, tasks, cols, ops=OPS) -> dict:
    """Leave-one-task-out: weights and thresholds from the other tasks only. Positive class: failure."""
    n = len(rows)
    score = np.full(n, np.nan)
    flags = {o: np.zeros(n, bool) for o in ops}
    weights = []
    for h in np.unique(tasks):
        te, tr = np.flatnonzero(tasks == h), np.flatnonzero(tasks != h)
        Xtr = _matrix([rows[i] for i in tr], cols, h)
        Xte = _matrix([rows[i] for i in te], cols, h)
        # A trial with no usable column at all (no same-task reference) is out of this detector.
        fit = np.isfinite(Xtr).any(1)
        str_, ste, w = _fit_score(Xtr[fit], y[tr][fit], Xte)
        weights.append(w)
        cov = np.isfinite(Xte).any(1)
        ste[~cov] = np.nan
        score[te] = ste
        succ = str_[~y[tr][fit]]
        for o in ops:
            th = np.quantile(succ, 1 - o) if len(succ) else np.inf
            flags[o][te] = np.nan_to_num(ste, nan=-np.inf) > th
    return dict(score=score, flags=flags, weights=weights)


def prf(flag, y, valid=None) -> dict:
    if valid is not None:
        flag, y = flag[valid], y[valid]
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    return dict(n=int(len(y)), fail=int(y.sum()), ok=int((~y).sum()), recall=tp / max(int(y.sum()), 1),
                far=fp / max(int((~y).sum()), 1), precision=(tp / (tp + fp)) if tp + fp else float('nan'))


def run_detectors(rows, y, tasks, arms, mags, detectors=DETECTORS) -> dict:
    out = {}
    for name, cols in detectors.items():
        res = loto(rows, y, tasks, cols)
        s = res['score']
        valid = np.isfinite(s)
        within = [auc(s[tasks == h], y[tasks == h]) for h in np.unique(tasks)]
        rec = dict(auc=auc(s, y), ci=_task_ci(s, y, tasks), auc_within_task=_nanmean(within),
                   auc_per_task={int(h): w for h, w in zip(np.unique(tasks), within)},
                   coverage=float(valid.mean()),
                   ops={str(o): prf(res['flags'][o], y, valid) for o in OPS},
                   by_arm={a: auc(s[arms == a], y[arms == a]) for a in np.unique(arms)},
                   by_mag={str(m): auc(s[mags == m], y[mags == m]) for m in np.unique(mags)},
                   far_by_arm={a: {str(o): prf(res['flags'][o], y, valid & (arms == a))['far'] for o in OPS}
                               for a in np.unique(arms)},
                   recall_by_arm={a: {str(o): prf(res['flags'][o], y, valid & (arms == a))['recall'] for o in OPS}
                                  for a in np.unique(arms)})
        rec['_score'], rec['_flags'] = s, res['flags']
        out[name] = rec
    return out


def _nanmean(x) -> float:
    x = np.array(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')


def cluster_task_ci(values, tasks, reps=5000, seed=0) -> tuple:
    """Mean of per-task means with a percentile CI from resampling tasks; also the leave-one-task-out range."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(tasks)
    per = np.array([values[tasks == u].mean() for u in uniq])
    boot = per[rng.integers(0, len(per), (reps, len(per)))].mean(1)
    lo = [float(np.delete(per, i).mean()) for i in range(len(per))]
    return float(per.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), min(lo), max(lo)


# tables

PAIRS = (('off', 'lr'), ('off_st', 'lr_st'), ('lr', 'on'), ('lr_st', 'on_st'), ('lr_st', 'lr'), ('on_st', 'on'),
         ('off_st', 'sd_st'), ('lr_st', 'lr_st_rf'), ('on_st', 'on_st_rf'))
ARMS = ('off', 'lr', 'on', 'off_st', 'sd_st', 'lr_st', 'on_st', 'lr_st_rf', 'on_st_rf')


def _subset(rows, trials_by_key, tag, keep) -> tuple:
    sub = [r for r in rows if r['tag'] == tag and keep(trials_by_key[r['key']])]
    ts = [trials_by_key[r['key']] for r in sub]
    return sub, ts, np.array([t['task'] for t in ts]), np.array([t['arm'] for t in ts]), np.array([t['mag'] for t in ts])


def _fmt_det(name, rec) -> str:
    o = rec['ops']
    return (f"{name:30s} {rec['auc']:.3f} [{rec['ci'][0]:.3f},{rec['ci'][1]:.3f}] within {rec['auc_within_task']:.3f} cov {rec['coverage']:.2f} | "
            + ' | '.join(f"FAR {o[str(x)]['far']:.3f} rec {o[str(x)]['recall']:.3f} prec {o[str(x)]['precision']:.3f}" for x in OPS))


def strategy_table(trials, refs, P) -> dict:
    """Distinct note sequences, labels and path clusters among successful last attempts, per task."""
    P('\n== Strategies among successful trials (last attempts) ==')
    P('task  n_ok  seqs(type+label)  seqs(type only)  pick_labels  clusters obj@1/2/4cm  world@1/2/4cm  largest obj clusters @2cm')
    out = {}
    for task in sorted({t['task'] for t in trials}):
        ok = [t for t in trials if t['task'] == task and t['success']]
        seqs = Counter(t['seqs'][-1] for t in ok)
        types = Counter(tuple(x[0] for x in t['seqs'][-1]) for t in ok)
        labs = Counter(next((l for _, l in t['seqs'][-1] if l), None) for t in ok)
        ro, rw = refs.get((task, -1, 'obj')), refs.get((task, -1, 'world'))
        rec = dict(n_ok=len(ok), seqs=len(seqs), types=len(types), labels=dict(labs),
                   obj=ro['counts'] if ro else {}, world=rw['counts'] if rw else {})
        out[task] = rec
        P(f"{task:4d} {len(ok):5d} {len(seqs):8d} {len(types):16d}  {dict(labs)}  obj {rec['obj']} world {rec['world']} sizes {(ro or {}).get('sizes', [])[:6]}")
    return out


def _cluster_meaning(trials, refs, P) -> None:
    """What the object-frame clusters are: each successful path against its nearest medoid."""
    P('\nWhat separates the two largest object-frame clusters (success paths assigned to the nearest of the 2 largest medoids)')
    for task in sorted({t['task'] for t in trials}):
        r = refs.get((task, -1, 'obj'))
        if not r or len(r['many']) < 2:
            continue
        ok = [t for t in trials if t['task'] == task and t['success']]
        tab: dict = defaultdict(Counter)
        for t in ok:
            d = [path_distance(t['paths'][-1]['obj'], q)[0] for q in r['many'][:2]]
            c = int(np.argmin(d))
            tab[c]['n'] += 1
            tab[c]['held_end'] += bool(t['attempts'][-1].get('gt_held_end'))
            tab[c]['has_release_gt20'] += t['attempts'][-1].get('ng_release_steps', 0) >= 20
            tab[c]['recovered'] += t['n_attempts'] > 1
            tab[c]['displaced'] += t['mag'] > 0
        P(f'  task {task}: ' + ' | '.join(f'c{c} {dict(v)}' for c, v in sorted(tab.items())))


def _per_task_far(det, y, tasks, names, P) -> None:
    P('\nHeld-out false alarms per task at nominal 5% (flagged share of true successes; n ok in the header)')
    P(' ' * 26, ' '.join(f'{h:>5d}' for h in np.unique(tasks)))
    P(f"{'n_ok':26s}", ' '.join(f'{int(((tasks == h) & ~y).sum()):5d}' for h in np.unique(tasks)))
    for name in names:
        fl = det[name]['_flags'][0.05]
        s_ = np.isfinite(det[name]['_score'])
        P(f'{name:26s}', ' '.join(f"{(fl[(tasks == h) & ~y & s_].mean() if ((tasks == h) & ~y & s_).any() else float('nan')):5.2f}" for h in np.unique(tasks)))


def _raw_threshold_loco(trials, refs, final_rows, final_ts, P) -> None:
    """A strategy absent from the references: threshold on raw DTW from other tasks' successes, then leave its cluster out."""
    P('\n== Valid strategy absent from the references: raw DTW threshold from other tasks, nominal 5% ==')
    for frame in ('obj', 'world'):
        for kind in ('one', 'many'):
            own_f, oth_f, n, fars = 0.0, 0.0, 0, []
            for h in range(10):
                tr = [r['r'][f'{frame}_{kind}'] for r, t in zip(final_rows, final_ts) if t['task'] != h and t['success']]
                tr = np.array(tr, float)
                tr = tr[np.isfinite(tr)]
                th = float(np.quantile(tr, 0.95))
                te = np.array([r['r'][f'{frame}_{kind}'] for r, t in zip(final_rows, final_ts) if t['task'] == h and t['success']], float)
                if te.size:
                    fars.append((float(np.nanmean(te > th)), te.size))
                r = refs.get((h, -1, frame))
                if r and r.get('leave_cluster_out') and kind == 'many':
                    a = np.array(r['leave_cluster_out'])
                    own_f += float((a[:, 0] > th).sum())
                    oth_f += float((a[:, 1] > th).sum())
                    n += len(a)
            far = sum(f * c for f, c in fars) / max(sum(c for _, c in fars), 1)
            msg = f'{frame:6s}{kind:5s} held-out FAR on successes {far:.3f}'
            if n:
                msg += f'; sampled paths n={n}: flagged with own cluster in the set {own_f / n:.3f}, own cluster removed {oth_f / n:.3f}'
            P(msg)


def report(trials, refs, scores, out: Path, refs_st=None, scores_st=None) -> None:
    for t in trials:
        rec = scores[_key(t)]
        for tag, a in (('first', 0), ('last', t['n_attempts'] - 1)):
            if 'd_stale' in rec[tag]:
                t['attempts'][a]['stale_dominant'] = bool(rec[tag]['d_stale'] < rec[tag]['d_true'])
        t['label'] = failure_label(t)
    by_key = {_key(t): t for t in trials}
    rows = build_rows(trials, scores)
    lines: list = []
    res: dict = {}

    def P(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    res['strategies'] = strategy_table(trials, refs, P)
    _cluster_meaning(trials, refs, P)

    # Held-out detection, at the end of the trial, every arm.
    final_rows, final_ts, tasks, arms, mags = _subset(rows, by_key, 'last', lambda t: True)
    y = np.array([not t['success'] for t in final_ts])
    P(f'\n== Held-out (leave-one-task-out) failure detection, last attempt, all arms, n={len(y)}, failures={int(y.sum())} ==')
    P('detector: pooled AUC [task bootstrap CI], mean within-task AUC (tasks with both classes), coverage | at training FAR 1%, 5%, 10% on training successes')
    det = run_detectors(final_rows, y, tasks, arms, mags)
    for name, rec in det.items():
        P(_fmt_det(name, rec))
    res['final'] = {k: {x: v for x, v in r.items() if not x.startswith('_')} for k, r in det.items()}
    P('\nPer-task held-out AUC of each main detector (nan: one class or no reference)')
    for name in ('cosine', 'dtw_obj_many_xtask', 'dtw_obj_many', 'cos+dtw_obj_many_xtask', 'cos+dist', 'cos+dist+dtw_many'):
        P(f"{name:26s}", ' '.join(f"{det[name]['auc_per_task'].get(h, float('nan')):.2f}" for h in range(10)))
    P('\nHeld-out AUC by arm')
    P(' ' * 30, ' '.join(f'{a:>8s}' for a in ARMS))
    for name in ('cosine', 'dtw_obj_many_xtask', 'dtw_obj_many', 'cos+dtw_obj_many_xtask', 'cos+dist', 'nogt+dtw_world'):
        P(f'{name:30s}', ' '.join(f"{det[name]['by_arm'].get(a, float('nan')):8.3f}" for a in ARMS))
    P('\nHeld-out AUC by magnitude (cm)')
    for name in ('cosine', 'dtw_obj_many_xtask', 'dtw_world_many', 'dtw_obj_many', 'cos+dtw_obj_many_xtask', 'cos+dist', 'nogt+dtw_world'):
        P(f'{name:30s}', ' '.join(f"{m}: {det[name]['by_mag'].get(m, float('nan')):.3f}" for m in ('0.0', '2.0', '5.0', '10.0')))

    # False alarms on true successes.
    P('\n== False alarms: truly successful trials flagged, held-out, thresholds from training tasks ==')
    P('detector, nominal FAR -> held-out FAR overall, then per arm at 5%')
    for name in ('cosine', 'dtw_world_one', 'dtw_world_many', 'dtw_obj_one', 'dtw_obj_many', 'dtw_obj_one_xtask', 'dtw_obj_many_xtask',
                 'cos+dtw_obj_many_xtask', 'cos+dist', 'nogt_only', 'nogt+dtw_world'):
        o = det[name]['ops']
        P(f"{name:26s} 1%->{o['0.01']['far']:.3f} 5%->{o['0.05']['far']:.3f} 10%->{o['0.1']['far']:.3f}  ",
          ' '.join(f"{a}:{det[name]['far_by_arm'].get(a, {}).get('0.05', float('nan')):.2f}" for a in ARMS))

    _per_task_far(det, y, tasks, ('cosine', 'dist_no_final', 'dtw_world_many', 'dtw_obj_one', 'dtw_obj_many', 'dtw_obj_many_xtask', 'cos+dist', 'nogt+dtw_world'), P)
    _raw_threshold_loco(trials, refs, final_rows, final_ts, P)
    # One reference against many, including when the path's own strategy is not in the reference set.
    P('\n== One reference vs many, leave-one-cluster-out ==')
    res['loo_cluster'] = {}
    for frame in ('obj', 'world'):
        own, oth = [], []
        for task in range(10):
            r = refs.get((task, -1, frame))
            if r and r.get('leave_cluster_out'):
                a = np.array(r['leave_cluster_out'])
                own.append(a[:, 0])
                oth.append(a[:, 1])
                res['loo_cluster'][f'{frame}_{task}'] = dict(n=len(a), own=float(np.median(a[:, 0])), others=float(np.median(a[:, 1])))
        if own:
            own, oth = np.concatenate(own), np.concatenate(oth)
            P(f'{frame}: paths in multi-cluster tasks n={len(own)}; DTW to own medoid median {100 * np.median(own):.2f} cm, p95 {100 * np.percentile(own, 95):.2f}; '
              f'to nearest other-cluster medoid median {100 * np.median(oth):.2f}, p95 {100 * np.percentile(oth, 95):.2f}; '
              f'over {CLUSTER_CM} cm: own {np.mean(own > CLUSTER_CM / 100):.3f}, other {np.mean(oth > CLUSTER_CM / 100):.3f}')

    # Timing in recovery arms: a recovered trial scored at each moment.
    P('\n== Recovered trials (success, more than one attempt): flagged at each time, nominal 5% ==')
    first_rows, first_ts, ftasks, farms, fmags = _subset(rows, by_key, 'first', lambda t: t['arm'] in ('off', 'off_st', 'sd_st', 'lr', 'on'))
    # Attempt truth exists here: the single-attempt arms, and sim-check arms where a second attempt means the first failed.
    y0 = np.array([not (t['success'] and t['n_attempts'] == 1) for t in first_ts])
    fdet = run_detectors(first_rows, y0, ftasks, farms, fmags, {k: DETECTORS[k] for k in ('cosine', 'dtw_obj_many_xtask', 'cos+dtw_obj_many_xtask', 'cos+dist', 'nogt+dtw_world')})
    P(f'Attempt 0 truth, arms off/off_st/sd_st/lr/on, n={len(y0)}, failed attempts={int(y0.sum())}')
    for name, rec in fdet.items():
        P(_fmt_det(name, rec))
    res['first'] = {k: {x: v for x, v in r.items() if not x.startswith('_')} for k, r in fdet.items()}
    whole_rows = [dict(key=k, tag='whole', r={'whole': scores[k].get('whole', float('nan')), '_xtask': None}) for k in (r['key'] for r in final_rows)]
    wdet = run_detectors(whole_rows, y, tasks, arms, mags, {'dtw_whole_world': ['whole']})
    P(_fmt_det('dtw_whole_world(naive, last-attempt truth)', wdet['dtw_whole_world']))
    res['whole'] = {k: v for k, v in wdet['dtw_whole_world'].items() if not k.startswith('_')}
    rec_mask = np.array([t['success'] and t['n_attempts'] > 1 for t in final_ts])
    fo_mask = np.array([t['success'] and t['n_attempts'] == 1 for t in final_ts])
    res['recovered_flag'] = {}
    for name in ('cosine', 'dtw_world_one', 'dtw_world_many', 'dtw_obj_many', 'dtw_obj_many_xtask', 'cos+dtw_obj_many_xtask', 'cos+dist'):
        fl = det[name]['_flags'][0.05]
        s = np.isfinite(det[name]['_score'])
        P(f"{name:28s} scored on last attempt: recovered flagged {fl[rec_mask & s].mean():.3f} (n={int((rec_mask & s).sum())}), first-try flagged {fl[fo_mask & s].mean():.3f} (n={int((fo_mask & s).sum())})")
    fl = wdet['dtw_whole_world']['_flags'][0.05]
    P(f"{'dtw_whole_world':28s} scored on whole trial: recovered flagged {fl[rec_mask].mean():.3f}, first-try flagged {fl[fo_mask].mean():.3f}")
    # The same recovered trials scored on their first attempt, which did fail.
    k2i = {r['key']: i for i, r in enumerate(first_rows)}
    for name in fdet:
        fl = fdet[name]['_flags'][0.05]
        ks = [i for i, t in enumerate(first_ts) if t['success'] and t['n_attempts'] > 1]
        if ks:
            P(f"{name:28s} scored at attempt 0 (trial later recovers): flagged {fl[ks].mean():.3f}, n={len(ks)}")

    # Labels.
    P('\n== Automatic labels against the rule classifier and simulator success ==')
    labels = sorted({t['label'] for t in trials})
    rules = sorted({t['rule'] for t in trials})
    P('auto label x success:', {l: [sum(t['label'] == l and t['success'] for t in trials), sum(t['label'] == l and not t['success'] for t in trials)] for l in labels}, '[sim success, sim fail]')
    ok_pred = np.array([t['label'] in ('in_goal', 'held_over_goal') for t in trials])
    ok_sim = np.array([t['success'] for t in trials])
    P(f'label in_goal or held_over_goal vs sim success: precision {(ok_pred & ok_sim).sum() / max(ok_pred.sum(), 1):.3f}, recall {(ok_pred & ok_sim).sum() / ok_sim.sum():.3f}, accuracy {(ok_pred == ok_sim).mean():.3f}')
    P('confusion on simulator failures (rows auto, cols rule):')
    fails = [t for t in trials if not t['success']]
    P(' ' * 24, ' '.join(f'{r[:12]:>12s}' for r in rules))
    conf: dict = {}
    for l in labels:
        c = Counter(t['rule'] for t in fails if t['label'] == l)
        conf[l] = dict(c)
        P(f'{l:24s}', ' '.join(f'{c.get(r, 0):12d}' for r in rules))
    res['confusion'] = conf
    P('\nFailure mode (auto) by arm, simulator failures only; success counts first')
    modes = sorted({t['label'] for t in fails})
    P(' ' * 10, 'first_ok  recov_ok', ' '.join(f'{m[:12]:>12s}' for m in modes))
    res['modes'] = {}
    for a in ARMS:
        ts = [t for t in trials if t['arm'] == a]
        c = Counter(t['label'] for t in ts if not t['success'])
        fo, ro = sum(t['success'] and t['n_attempts'] == 1 for t in ts), sum(t['success'] and t['n_attempts'] > 1 for t in ts)
        res['modes'][a] = dict(first_ok=fo, recovered_ok=ro, **{m: c.get(m, 0) for m in modes})
        P(f'{a:10s} {fo:8d} {ro:9d}', ' '.join(f'{c.get(m, 0):12d}' for m in modes))

    # Paired mechanism.
    P('\n== Paired conversions (same task, phase, magnitude, trial) ==')
    res['pairs'] = {}
    for a, b in PAIRS:
        ka = [k for k in by_key if k[0] == a]
        diff = np.array([float(by_key[(b, *k[1:])]['success']) - float(by_key[k]['success']) for k in ka])
        tk = np.array([k[1] for k in ka])
        mean, lo, hi, mn, mx = cluster_task_ci(diff, tk)
        gains = [k for k in ka if not by_key[k]['success'] and by_key[(b, *k[1:])]['success']]
        losses = [k for k in ka if by_key[k]['success'] and not by_key[(b, *k[1:])]['success']]
        P(f'\n{a} -> {b}: {100 * mean:+.1f} points [task CI {100 * lo:+.1f},{100 * hi:+.1f}; leave-one-task-out {100 * mn:+.1f}..{100 * mx:+.1f}]; {len(gains)} gained, {len(losses)} lost of {len(ka)}')
        tab: dict = defaultdict(Counter)
        for k in gains:
            tab[by_key[k]['label']][k[1]] += 1
        P('  converted, by mode in A, then by task:')
        for m, c in sorted(tab.items(), key=lambda x: -sum(x[1].values())):
            tot = sum(1 for k in ka if not by_key[k]['success'] and by_key[k]['label'] == m)
            P(f'    {m:24s} {sum(c.values()):4d}/{tot:4d}  {dict(sorted(c.items()))}')
        ltab: dict = defaultdict(Counter)
        for k in losses:
            ltab[by_key[(b, *k[1:])]['label']][k[1]] += 1
        P('  lost, by mode in B, then by task:')
        for m, c in sorted(ltab.items(), key=lambda x: -sum(x[1].values())):
            P(f'    {m:24s} {sum(c.values()):4d}  {dict(sorted(c.items()))}')
        by_task = {int(t): [int(sum(by_key[k]['success'] for k in ka if k[1] == t)), int(sum(by_key[(b, *k[1:])]['success'] for k in ka if k[1] == t))] for t in np.unique(tk)}
        P('  successes A,B by task:', by_task)
        res['pairs'][f'{a}|{b}'] = dict(mean=mean, ci=[lo, hi], loto=[mn, mx], gained=len(gains), lost=len(losses), by_task=by_task,
                                        converted={m: dict(c) for m, c in tab.items()}, lost_modes={m: dict(c) for m, c in ltab.items()})

    # Silent failures: strict arms, no recovery recorded.
    P('\n== Silent failures: strict arms, failed, no recovery attribution ==')
    res['silent'] = {}
    for a in ('lr_st', 'on_st'):
        ts = [t for t in trials if t['arm'] == a and not t['success'] and t['n_recovery'] == 0]
        all_fail = sum(1 for t in trials if t['arm'] == a and not t['success'])
        P(f'{a}: silent {len(ts)} of {all_fail} failures')
        for task in range(10):
            sub = [t for t in ts if t['task'] == task]
            if not sub:
                continue
            P(f"  task {task} {sub[0]['name'][12:36]:24s} n={len(sub):3d} labels {dict(Counter(t['label'] for t in sub).most_common())} "
              f"outcomes {dict(Counter(tuple(t['outcomes'])[:1] for t in sub).most_common(2))} "
              f"lifted {dict(Counter(tuple(n.replace('_1_main', '') for n in t['attempts'][0]['lifted']) for t in sub).most_common(3))}")
        res['silent'][a] = {str(task): dict(Counter(t['label'] for t in ts if t['task'] == task)) for task in range(10)}
    P('\nFirst close, per task, 0 cm and all arms: nearest non-goal body and pick label')
    for task in (0, 3, 4, 6, 8):
        sub = [t for t in trials if t['task'] == task and t['arm'] == 'off_st']
        c = Counter((t['pick_label'], (t['attempts'][0].get('nearest_at_close') or 'none').replace('_1_main', ''), t['success']) for t in sub)
        P(f'  task {task}: {c.most_common(8)}')

    # Done-flag artifact: release cut short, object still in the gripper.
    P('\n== Release cut short (under 20 steps) and object still held at the end, by arm ==')
    res['held_end'] = {}
    for a in ARMS:
        ts = [t for t in trials if t['arm'] == a]
        short = [t for t in ts if t['attempts'][-1].get('ng_release_steps', 0) < 20]
        held = [t for t in ts if t['attempts'][-1].get('gt_held_end')]
        res['held_end'][a] = dict(short=len(short), short_ok=sum(t['success'] for t in short), held=len(held), held_ok=sum(t['success'] for t in held),
                                  ok=sum(t['success'] for t in ts))
        P(f"{a:10s} short {len(short):3d} (ok {sum(t['success'] for t in short):3d}) held at end {len(held):3d} (ok {sum(t['success'] for t in held):3d}) of ok {sum(t['success'] for t in ts):3d}")
    P('Held at end by task, lr_st:', dict(sorted(Counter((t['task'], t['success']) for t in trials if t['arm'] == 'lr_st' and t['attempts'][-1].get('gt_held_end')).items())))

    # Butter.
    P('\n== Butter: what each attempt lifted, by arm, all magnitudes ==')
    res['butter'] = {}
    for a in ('lr', 'on', 'lr_st', 'on_st'):
        ts = [t for t in trials if t['task'] == 6 and t['arm'] == a]
        for mag in (0.0, 2.0, 5.0, 10.0):
            sub = [t for t in ts if t['mag'] == mag]
            seq = Counter(tuple((x['lifted'][0].replace('_1_main', '') if x['lifted'] else '-') for x in t['attempts']) for t in sub)
            P(f"  {a:6s} {mag:4g} cm ok {sum(t['success'] for t in sub):2d}/{len(sub)} attempts {dict(Counter(t['n_attempts'] for t in sub))} lifted-per-attempt {seq.most_common(5)}")
        res['butter'][a] = sum(t['success'] for t in ts)

    P('\nButter: the attempt where butter first reached the basket, its end state, and the basket aim')
    for a in ('lr', 'on'):
        vals = defaultdict(list)
        for t in trials:
            if t['task'] != 6 or t['arm'] != a:
                continue
            for i, f in enumerate(t['attempts']):
                if 'butter_1_main' in f['lifted']:
                    tag = 'accepted' if i == t['n_attempts'] - 1 and t['success'] else 'not_accepted'
                    vals[tag].append((f['gt_goal_xy'], f['gt_goal_dz'], i))
                    break
        for tag, v in vals.items():
            v = np.array(v)
            P(f'  {a:3s} {tag:13s} n={len(v):2d} goal_xy cm median {100 * np.median(v[:, 0]):.1f} range {100 * v[:, 0].min():.1f}-{100 * v[:, 0].max():.1f}; z over basket origin cm median {100 * np.median(v[:, 1]):.1f} range {100 * v[:, 1].min():.1f}-{100 * v[:, 1].max():.1f}; attempt {dict(Counter(v[:, 2].astype(int).tolist()))}')
        P(f"  {a:3s} retargets per trial: {np.mean([t['n_retarget'] for t in trials if t['task'] == 6 and t['arm'] == a]):.2f}")
    P('Recoveries on the other strict-zero tasks, arm lr, what each attempt lifted:')
    for task in (3, 4, 8, 0):
        c = Counter(tuple((f['lifted'][0].replace('_1_main', '') if f['lifted'] else '-') for f in t['attempts']) + (t['success'],) for t in trials if t['arm'] == 'lr' and t['task'] == task)
        P(f'  task {task}: {c.most_common(4)}')

    # Stale against shifted reference, the source-against-destination dominance of the injection study.
    P('\n== Approach path: nearer the same-seed 0 cm path (stale) or that path shifted by the displacement (true) ==')
    res['dominance'] = {}
    for a in ARMS:
        vals = {}
        for mag in (2.0, 5.0, 10.0):
            for tag, ix in (('first', 0), ('last', -1)):
                sub = [t for t in trials if t['arm'] == a and t['mag'] == mag]
                st = [t['attempts'][ix].get('stale_dominant') for t in sub]
                st = [s for s in st if s is not None]
                vals[f'{tag}@{mag:g}'] = float(np.mean(st)) if st else float('nan')
        res['dominance'][a] = vals
        P(f'{a:10s} stale-dominant share', {k: round(v, 3) for k, v in vals.items()})

    # Online, no ground truth: strict trials the robot's own checks accepted (no recovery started).
    P('\n== Online estimate: replace the sim check with a no-ground-truth detector ==')
    strict = ('off_st', 'sd_st', 'lr_st', 'on_st', 'lr_st_rf', 'on_st_rf')
    on_rows, on_ts, otasks, oarms, omags = _subset(rows, by_key, 'last', lambda t: t['arm'] in strict and t['n_attempts'] == 1)
    oy = np.array([not t['success'] for t in on_ts])
    P(f'Population: strict arms, robot accepted after one attempt, n={len(oy)}, silent failures={int(oy.sum())}')
    odet = {k: DETECTORS[k] for k in ('nogt_no_refs', 'nogt_only', 'nogt+dtw_world', 'nogt+dtw_world+done_leak', 'cos+dist', 'cos+dist+dtw_many')}
    odet['aim_replaces_true'] = ['ng_cos_aim', 'ng_cos_aim_late', 'ng_cos_aim_carry', 'ng_aim_gap'] + _dtw_cols('aim', 'many')
    orec = run_detectors(on_rows, oy, otasks, oarms, omags, odet)
    if scores_st is not None:
        # References and jaw widths from strict arms alone: no trial the sim check rescued feeds the library.
        st_trials = [t for t in trials if t['arm'] in strict]
        rows_st = build_rows(trials, scores_st, ref_trials=st_trials)
        st_rows, _, st_tasks, st_arms, st_mags = _subset(rows_st, by_key, 'last', lambda t: t['arm'] in strict and t['n_attempts'] == 1)
        assert [r['key'] for r in st_rows] == [r['key'] for r in on_rows]
        orec2 = run_detectors(st_rows, oy, st_tasks, st_arms, st_mags, {'strict_refs:nogt+dtw_world': DETECTORS['nogt+dtw_world'], 'strict_refs:aim': odet['aim_replaces_true']})
        orec.update(orec2)
    for name, rec in orec.items():
        P(_fmt_det(name, rec))
    res['online'] = {k: {x: v for x, v in r.items() if not x.startswith('_')} for k, r in orec.items()}
    idx = {r['key']: i for i, r in enumerate(on_rows)}
    res['online_gain'] = {}
    for st_arm, sim_arm in (('lr_st', 'lr'), ('on_st', 'on')):
        ka = [k for k in by_key if k[0] == st_arm]
        base = np.array([by_key[k]['success'] for k in ka], float)
        sim = np.array([by_key[(sim_arm, *k[1:])]['success'] for k in ka], float)
        accepted = np.array([by_key[k]['n_attempts'] == 1 for k in ka])
        tk = np.array([k[1] for k in ka])
        mag = np.array([k[3] for k in ka])
        P(f'\n{st_arm} {100 * base.mean():.1f} -> {sim_arm} {100 * sim.mean():.1f}: sim-check advantage {100 * (sim - base).mean():+.1f} points (displaced only {100 * (sim - base)[mag > 0].mean():+.1f}, 10 cm {100 * (sim - base)[mag == 10].mean():+.1f})')
        perfect = np.where(accepted & (base == 0), sim, base)
        P(f'  perfect detector, flags only accepted trials that failed: {100 * (perfect - base).mean():+.1f} points')
        for name in orec:
            for o in OPS:
                fl = np.zeros(len(ka), bool)
                for i, k in enumerate(ka):
                    if k in idx:
                        fl[i] = orec[name]['_flags'][o][idx[k]]
                optimistic = np.where(fl, sim, base)                # a flagged success stays a success if the sim-check arm succeeded
                pessimistic = np.where(fl & (base == 1), 0.0, optimistic)  # a flagged success is lost
                g, glo, ghi, _, _ = cluster_task_ci(optimistic - base, tk)
                gp, plo, phi, _, _ = cluster_task_ci(pessimistic - base, tk)
                P(f"  {name:26s} @{o:4.2f}: flagged {fl.mean():.3f}, gain if false alarms harmless {100 * g:+.1f} [{100 * glo:+.1f},{100 * ghi:+.1f}], if they lose the trial {100 * gp:+.1f} [{100 * plo:+.1f},{100 * phi:+.1f}]")
                res['online_gain'][f'{st_arm}|{name}|{o}'] = dict(flag=float(fl.mean()), opt=g, opt_ci=[glo, ghi], pess=gp, pess_ci=[plo, phi])

    # Sample to render, per label.
    P('\n== Trials to render for the spot check (no frames were saved) ==')
    rng = np.random.default_rng(0)
    res['render'] = {}
    for l in sorted({t['label'] for t in trials}):
        sub = [t for t in trials if t['label'] == l]
        take = [sub[i] for i in rng.choice(len(sub), min(5, len(sub)), replace=False)]
        res['render'][l] = [list(_key(t)) for t in take]
        P(f'{l}: ' + '; '.join(f"{t['arm']}/task{t['task']}/{trace_stem(t['phase'], t['mag'], t['trial'])}(sim {'ok' if t['success'] else 'fail'}, rule {t['rule']})" for t in take))

    (out / 'tables.txt').write_text('\n'.join(lines) + '\n')
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=str))
    per = [{**dict(zip(('arm', 'task', 'phase', 'mag', 'trial'), _key(t))), 'success': t['success'], 'n_attempts': t['n_attempts'], 'label': t['label'], 'rule': t['rule']} for t in trials]
    (out / 'labels.json').write_text(json.dumps(per))
    cols = ['gt_cos_obj', 'gt_cos_obj_late', 'gt_cos_goal', 'ng_cos_aim', 'gt_closing', 'gt_close_xy', 'gt_goal_xy', 'gt_goal_dz']
    with open(out / 'signals.csv', 'w') as fh:
        fh.write(','.join(['arm', 'task', 'phase', 'mag', 'trial', 'success', 'n_attempts', 'label', 'rule', *cols, 'dtw_world_many', 'dtw_obj_many']) + '\n')
        for row in rows:
            if row['tag'] == 'last':
                t = by_key[row['key']]
                vals = [t['attempts'][-1].get(c, float('nan')) for c in cols] + [row['r']['world_many'], row['r']['obj_many']]
                fh.write(','.join(map(str, [*row['key'], t['success'], t['n_attempts'], t['label'], t['rule'], *(round(float(v), 4) for v in vals)])) + '\n')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dyn-root', required=True)
    ap.add_argument('--trace-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--diversity-only', action='store_true', help='print the strategies table and stop (partial runs)')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'features.pkl'
    if cache.exists():
        trials, refs, scores, refs_st, scores_st = pickle.loads(cache.read_bytes())
    else:
        trials = load_all(a.dyn_root, a.trace_root, a.workers)
        print(f'{len(trials)} trials', flush=True)
        if a.diversity_only:
            refs = build_references(trials, a.workers)
            strategy_table(trials, refs, print)
            _cluster_meaning(trials, refs, print)
            return
        refs = build_references(trials, a.workers)
        print(f'{len(refs)} reference sets', flush=True)
        scores = score_all(trials, refs, a.workers)
        strict = [t for t in trials if t['arm'].endswith(('_st', '_st_rf'))]
        refs_st = build_references(strict, a.workers)
        scores_st = score_all(trials, refs_st, a.workers)
        cache.write_bytes(pickle.dumps((trials, refs, scores, refs_st, scores_st), protocol=4))
    report(trials, refs, scores, out, refs_st, scores_st)


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
