"""Restrict which leaf skills the planner sees and the executor runs. Off by default.

SPARK_SKILLSET selects the arm (run_item.sh tokens _skgen, _skbase, _skplain, _skshape, _skask, _skaskw):
  generic  base primitives, grasp_se3 and the verify_* conditions; the shipped prompt minus every passage
           about a withheld skill.
  base     exactly the 5 base primitives, plus guidance on composing them (_BASE_COMPOSE).
  plain    base, and move_to_keypoint / grasp also drop their label-specific pick and place branches.
  shape    plain, and those branches are chosen and sized from the mask's world points (an object wider than
           the open jaws is pinched at its rim). Same prompt as base.
  ask      shape, but the planner chooses the rim pinch: base prompt plus one optional move_to_keypoint
           parameter, grasp_style.
  askw     ask, but grasp_style is required on every pick and the planner reads each detection's measured
           width (the same mask measurement as shape) in the detection details.
Unset: the shipped prompt is returned unchanged and nothing is withheld.

The variant is derived from the shipped text by anchored cuts. An anchor that no longer matches raises, so a
change to a shipped prompt cannot silently change an arm.
A v2 prompt (SPARK_PROMPT=v2, system_<family>_v2.txt) marks its skill-specific passages instead, and an arm
keeps a passage only when it allows every skill the passage names.
"""
from __future__ import annotations

import os
import re

from spark_core import bt_grammar

BASE = frozenset({'move_to_keypoint', 'move_relative', 'grasp', 'release', 'wait'})
GENERIC = BASE | {'grasp_se3', 'verify_grasp', 'verify_placed'}
_ALLOWED = {'generic': GENERIC, 'base': BASE, 'plain': BASE, 'shape': BASE, 'ask': BASE, 'askw': BASE}
_CONTROL = {'sequence', 'selector', 'fallback', 'retry', 'parallel', 'sync_barrier'}

# (first line prefix, last line prefix), inclusive: passages that name a withheld skill.
_CUTS = (
    ('- For UPRIGHT cylindrical objects', '- For UPRIGHT cylindrical objects'),
    ('- Seating an object into a socket', '  on. Do not ask for a "press" skill'),
    ('Use `place_in_slot` for a genuinely slotted tray', 'Use `place_in_slot` for a genuinely slotted tray'),
    ('- REQUIRED placement primitive for SLOTTED', '    - grasp_se3(spoon 1) -> move_relative(dz:0.20) -> move_to_keypoint(tray'),
    ('- (POUR TASKS', '- For "open drawer and put X inside"'),
    ('# SWEEP PRIMITIVE', '# END SOAP BOTTLE HANDLING'),
    ('POUR PRIMITIVE (vision-terminated):', '  splashout; raise (0.85)'),
    ('Recovery skills you may put in a branch', '- `compliant_push(...)`'),
)
_BASE_CUTS = (
    ('- For elongated or oddly shaped objects', '- For objects that need angled/side grasps'),
    ('- grasp_se3 replaces the move_to_keypoint', '- grasp_se3 replaces the move_to_keypoint'),
    ('- Sequence for SE(3) grasp', '- Sequence for SE(3) grasp'),
    ('Condition leaves (no motion;', '  can see the result, so it never triggers a blind retry.'),
)
_BASE_EDITS = (
    ('Sequence: grasp_se3(X) -> move_relative', 'Sequence: move_to_keypoint(X) -> grasp -> move_relative'),
    ('Acquire-then-verify inside a `retry`,\nplace-then-verify inside a `fallback`',
     'Acquire inside a `retry`,\nplace inside a `fallback`'),
    ('# acquire, verified, bounded', '# acquire, bounded'),
    ('# place, verified, one recovery', '# place, one recovery'),
    ('# ran ONLY if the place was not confirmed', '# the second branch'),
)

_ONLY_LISTED = """
SKILL SET: only the node types listed under "Available primitives" exist. Do not emit any other leaf type;
the executor drops it without running it.
"""

# How to compose higher-level skills from the base primitives.
_BASE_COMPOSE = """
BASE-PRIMITIVE COMPOSITION:
Higher-level skills can be built out of ONLY the base primitives
(move_to_keypoint, move_relative, grasp, release, wait). First DESCRIBE the
motion as a short sequence of base primitives, THEN emit the behavior tree.
If the composed skill for a verb (scrub / wipe / sweep / polish / stir) is not
in the Available primitives list, DECOMPOSE it into base primitives instead of
emitting a single high-level node.

Worked examples (base primitives only):
- Sweep debris toward an edge: move behind the pile, press down, then repeat
  move_relative in ONE direction (e.g. {dy: -0.08}) several times to push it off.
- Stack A on B without a `stack` node: move_to_keypoint(A) -> grasp ->
  move_relative {dz: 0.20} -> move_to_keypoint(B, offset_z: 0.06) -> release.
"""


_NEEDS = '@@needs '   # v2 prompts: `@@needs skill ...` opens a passage about those skills, `@@end` closes it

# ask arm: the one line of the catalogue that changes, and the passage appended to the base prompt.
_ASK_PARAMS = '  params: keypoint_label (str), offset_x (float), offset_y (float), offset_z (float)'
_ASK_STYLE = """
GRASP STYLE: when move_to_keypoint goes to an object to pick it up, the optional grasp_style chooses how the
jaws take it. "center" (default) closes them around the whole object from above. Use "rim" when the image shows
an object wider than the gripper's maximum opening (about 8 cm): the jaws then pinch its edge.
"""
_ASKW_STYLE = """
GRASP STYLE (required): every move_to_keypoint that goes to an object to pick it up sets grasp_style. It is a
separate parameter from grasp_strategy. The gripper opens 8 cm at most. Find the line of the object in the
detection details and read its width. A width above 8 cm (for example width=12.0cm) does not fit between the
jaws: write grasp_style: "rim", and the jaws pinch the edge of the object. A width of 8 cm or less (for
example width=6.0cm) fits: write grasp_style: "center", and the jaws close around the whole object. Write
"center" when the line has no width.
"""


def active() -> str:
    """The selected arm, '' when off. An unknown value raises rather than running the full library."""
    arm = os.environ.get('SPARK_SKILLSET', '').strip()
    if arm and arm not in _ALLOWED:
        raise ValueError(f'SPARK_SKILLSET={arm!r}; expected one of {sorted(_ALLOWED)}')
    return arm


def plain_pick() -> bool:
    """The label-specific pick and place branches are off."""
    return active() in ('plain', 'shape', 'ask', 'askw')


def geometry_pick() -> str:
    """'shape' or 'ask' when the pick and place branches come from the mask's points, else ''. The executor
    runs askw as ask."""
    arm = active()
    return {'shape': 'shape', 'ask': 'ask', 'askw': 'ask'}.get(arm, '')


def planner_widths() -> bool:
    """askw: the detection details carry each object's measured width."""
    return active() == 'askw'


def is_withheld(node_type) -> bool:
    arm = active()
    return bool(arm) and node_type not in _ALLOWED[arm] and node_type not in _CONTROL


def withheld_leaves(tree) -> list:
    """Leaf types in a plan tree that the arm withholds, depth-first."""
    if not active():
        return []
    return [n['type'] for n in bt_grammar.walk(tree) if n.get('type') and is_withheld(n['type'])]


def _cut(lines: list, first: str, last: str) -> list:
    i = next((k for k, s in enumerate(lines) if s.startswith(first)), None)
    if i is None:
        raise ValueError(f'frozen prompt changed: no line starts with {first!r}')
    j = next((k for k in range(i, len(lines)) if lines[k].startswith(last)), None)
    if j is None:
        raise ValueError(f'frozen prompt changed: no line starts with {last!r} after {first!r}')
    return lines[:i] + lines[j + 1:]


def _catalogue(lines: list, allowed) -> list:
    """Keep only the allowed entries of the "Available primitives" list (an entry and its params line)."""
    start = lines.index('Available primitives:') + 1
    end = lines.index('Output format (YAML):')
    kept, keep = [], False
    for s in lines[start:end]:
        if s.startswith('- '):
            keep = s[2:].split(':', 1)[0] in allowed
        if keep or not s.strip():
            kept.append(s)
    return lines[:start] + kept + lines[end:]


def _drop_yaml_nodes(text: str, types) -> str:
    """Remove `- type: X` nodes and their one params line from the example trees."""
    for t in types:
        text, n = re.subn(rf'^ *- type: {t}\n *params:.*\n', '', text, flags=re.M)
        if not n:
            raise ValueError(f'frozen prompt changed: no {t} node in the example tree')
    return text


def _marked(text: str, allowed) -> str:
    """A v2 prompt marks its skill-specific passages: a `@@needs a b` ... `@@end` block stays only when the
    arm allows every named skill (always when ``allowed`` is None). The marker lines are dropped."""
    out, keep = [], True
    for s in text.split('\n'):
        if s.startswith(_NEEDS):
            keep = allowed is None or set(s.split()[1:]) <= allowed
        elif s == '@@end':
            keep = True
        elif keep:
            out.append(s)
    return '\n'.join(out)


def restrict_prompt(text: str) -> str:
    """The planner system prompt for the active arm; ``text`` itself when the ablation is off."""
    arm = active()
    if _NEEDS in text:
        out = _marked(text, _ALLOWED[arm] if arm else None)
        return _arm_notes(out, arm) if arm else out
    if not arm:
        return text
    lines = _catalogue(text.split('\n'), _ALLOWED[arm])
    for first, last in _CUTS + (_BASE_CUTS if arm != 'generic' else ()):
        lines = _cut(lines, first, last)
    if arm != 'generic' and any(s.startswith('UR10e ARM MODE') for s in lines):
        lines = _cut(lines, 'UR10e ARM MODE', '  for grasp_se3.')
    out = '\n'.join(lines)
    if arm == 'generic':
        out = _drop_yaml_nodes(out, ['search_keypoint'])
    else:
        out = _drop_yaml_nodes(out, ['verify_grasp', 'verify_placed', 'search_keypoint'])
        for old, new in _BASE_EDITS:
            if old not in out:
                raise ValueError(f'frozen prompt changed: {old!r}')
            out = out.replace(old, new)
    return _arm_notes(out, arm)


def _arm_notes(out: str, arm: str) -> str:
    """Append the arm's notes; the ask arm also gets its one extra parameter."""
    out = out.rstrip('\n') + '\n' + _ONLY_LISTED + (_BASE_COMPOSE if arm != 'generic' else '')
    if arm in ('ask', 'askw'):
        if out.count(_ASK_PARAMS + '\n') != 1:
            raise ValueError('frozen prompt changed: move_to_keypoint params line')
        param, style = ((', grasp_style ("rim" | "center", optional)', _ASK_STYLE) if arm == 'ask'
                        else (', grasp_style ("rim" | "center")', _ASKW_STYLE))
        out = out.replace(_ASK_PARAMS + '\n', _ASK_PARAMS + param + '\n') + style
    return out


if __name__ == '__main__':   # python -m spark_core.skillset [family]: print the active arm's prompt for review
    import sys
    from pathlib import Path
    family = sys.argv[1] if len(sys.argv) > 1 else 'ur10e'
    print(restrict_prompt((Path(__file__).parent / 'prompts' / f'system_{family}.txt').read_text()))
