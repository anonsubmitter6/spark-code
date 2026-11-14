#!/usr/bin/env python3
"""Paired results of the skill-set ablation. Usage: skillset_summary.py <out_root>

Each item directory <suite>_<pert>_<task>_<off>_<n>_<arm> holds <suite>.json. Pairs are (setting, task, init state)
against the full library of the same protocol. Reports per-setting success, an exact two-sided sign test on the
discordant pairs and a percentile bootstrap over tasks, then per-task rates, task families and plan statistics.
"""
import collections
import glob
import json
import math
import os
import random
import re
import sys

import yaml

ARM_SUFFIX = re.compile(r'_sk(gen|base|plain)$')
ARMS = ('full', 'gen', 'base', 'plain')
FAMILY = (('drawer', r'\b(open|close)\b.*(drawer|layer)'), ('knob', r'\bturn\b'), ('push', r'\bpush\b'))


def family(instruction):
    il = instruction.lower()
    return next((f for f, rx in FAMILY if re.search(rx, il)), 'pick-place')


def leaves(y):
    out = []

    def walk(n):
        if isinstance(n, dict):
            if n.get('type') and not n.get('children'):
                out.append(n['type'])
            for c in n.get('children') or []:
                walk(c)
    try:
        walk((yaml.safe_load(y) or {}).get('tree', {}))
    except yaml.YAMLError:
        pass
    return out


def load(root):
    """{(protocol, arm): {(suite, pert, task, init): success}}, task instructions, plan counters."""
    res, instr, plans = collections.defaultdict(dict), {}, collections.defaultdict(collections.Counter)
    meta = collections.defaultdict(collections.Counter)
    for d in glob.glob(os.path.join(root, '*')):
        name = os.path.basename(d)
        parts = name.split('_', 5)
        if len(parts) != 6 or not parts[2].isdigit():
            continue
        suite, pert, task, off, _, arm = parts
        path = os.path.join(d, f'{suite}.json')
        if not os.path.exists(path):
            continue
        m = ARM_SUFFIX.search(arm)
        ablation, protocol = (m.group(1) if m else 'full'), ARM_SUFFIX.sub('', arm)
        for det in json.load(open(path))['perturbations'][pert]['task_details']:
            key = (suite, pert, int(task))
            instr[key] = det['instruction']
            for i, tm in enumerate(det['trial_meta']):
                res[protocol, ablation][(*key, int(off) + i)] = bool(tm['success'])
                plans[protocol, ablation].update(leaves(tm.get('bt_yaml') or ''))
                meta[protocol, ablation].update(n=1, plan_missing=not tm.get('bt_yaml'),
                                                retried='skillset_retry' in (tm.get('bt_yaml') or ''),
                                                leaf_skipped=bool(tm.get('skillset_skipped')),
                                                error='error' in tm)
    return res, instr, plans, meta


def sign_test_p(a, b):
    n = a + b
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n)


def bootstrap_ci(pairs, seed=0, reps=4000):
    """95% percentile CI of the mean paired difference, resampling (setting, task) clusters."""
    by = collections.defaultdict(list)
    for key, d in pairs:
        by[key[:3]].append(d)
    tasks = list(by)
    rng, means = random.Random(seed), []
    for _ in range(reps):
        s = [d for t in rng.choices(tasks, k=len(tasks)) for d in by[t]]
        means.append(sum(s) / len(s))
    means.sort()
    return means[int(.025 * reps)], means[int(.975 * reps)]


def main(root):
    res, instr, plans, meta = load(root)
    for protocol in sorted({p for p, _ in res}):
        print(f'\n=== protocol {protocol}')
        full = res.get((protocol, 'full'), {})
        for arm in ARMS:
            r = res.get((protocol, arm))
            if not r:
                continue
            print(f'\n-- {arm}: {sum(r.values())}/{len(r)} = {100 * sum(r.values()) / len(r):.1f}%  plans: '
                  f'{dict(meta[protocol, arm])}')
            print('   leaves:', plans[protocol, arm].most_common(12))
            cells = sorted({k[:2] for k in r})
            for cell in cells + [None]:
                keys = [k for k in r if cell is None or k[:2] == cell]
                paired = [k for k in keys if k in full]
                line = f'   {"/".join(cell) if cell else "ALL":<18} {sum(r[k] for k in keys):>3}/{len(keys):<3}'
                if arm != 'full' and paired:
                    a = sum(full[k] and not r[k] for k in paired)
                    b = sum(r[k] and not full[k] for k in paired)
                    diffs = [(k, int(r[k]) - int(full[k])) for k in paired]
                    lo, hi = bootstrap_ci(diffs)
                    line += (f' paired n={len(paired)} full {sum(full[k] for k in paired)} vs {sum(r[k] for k in paired)}'
                             f' diff {100 * (b - a) / len(paired):+.1f} pts [{100 * lo:+.1f},{100 * hi:+.1f}]'
                             f' lost {a} gained {b} sign p={sign_test_p(a, b):.2g}')
                print(line)
        print('\n-- per task and family (success counts: full gen base plain)')
        fams = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
        for key in sorted(instr):
            row = []
            for arm in ARMS:
                r = res.get((protocol, arm), {})
                vals = [v for k, v in r.items() if k[:3] == key]
                row.append(f'{sum(vals)}/{len(vals)}' if vals else '-')
                fams[(key[0], family(instr[key]))][arm][0] += sum(vals)
                fams[(key[0], family(instr[key]))][arm][1] += len(vals)
            print(f'   {key[0][:4]} {key[1][:3]} {key[2]} {family(instr[key]):<10} {" ".join(f"{c:>5}" for c in row)}  '
                  f'{instr[key][:58]}')
        for fam in sorted(fams):
            print('   family', fam, ' '.join(f'{a}={fams[fam][a][0]}/{fams[fam][a][1]}' for a in ARMS))


if __name__ == '__main__':
    main(sys.argv[1])
