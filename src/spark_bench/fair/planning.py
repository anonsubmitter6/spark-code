"""The runner's planner call and the post-parse label validator."""
from __future__ import annotations

import logging
from typing import Optional

import yaml

from spark_core.bt_validator import _LABEL_KEYS_STR, _iter_action_params, validate_and_repair_bt
from spark_bench.libero_pro.bddl import _OBJ_VISUAL_PROMPTS
from spark_bench.fair.config import FairConfig
from spark_bench.libero_pro.planning import _scripted_plan, plan_llm

log = logging.getLogger(__name__)


def _plan(cfg: FairConfig, instruction: str, det_map: dict, rgb, pick_hint: str, place_hint: str,
          picks: Optional[list] = None, places: Optional[list] = None, pick_counts: Optional[list] = None,
          prompts: Optional[list] = None):
    """The LLM plan, or the scripted plan under --no-gemini. ``picks`` / ``places`` / ``pick_counts``
    carry compound goals ("put both X and Y in Z")."""
    replay = getattr(cfg, '_replay_plans', None)
    if replay is not None:  # dyn --plan-replay: the plans a recorded trial wrote, in order; no LLM call
        return yaml.safe_load(replay.pop(0)) if replay else None
    labels = list(det_map.keys())
    if cfg.no_gemini:
        return _scripted_plan(labels, pick_hint, place_hint, instruction,
                              picks=picks, places=places, det_map=det_map, prompts=prompts)
    return plan_llm(instruction, labels, pick_hint, place_hint, det_map=det_map, rgb_image=rgb,
                    no_bddl_hints=cfg.no_bddl_hints, picks=picks, places=places, pick_counts=pick_counts)


def same_object_phrases(label: str) -> list[str]:
    """The detection phrases of the object ``label`` names: by object name (cream_cheese, 'cream cheese') or
    by being one of its phrases."""
    low = label.lower()
    words = low.replace('_', ' ').split()
    for obj, phrases in _OBJ_VISUAL_PROMPTS.items():
        if words == obj.split('_') or low in (p.lower() for p in phrases):
            return list(phrases)
    return []


def _label_params(score: dict):
    """(params, key, value) of every label parameter in the plan."""
    for params in _iter_action_params(score):
        for key in _LABEL_KEYS_STR:
            yield params, key, params.get(key)


def snap_to_object_phrase(score: dict, det_map: dict) -> list[dict]:
    """In place: a plan label outside ``det_map`` that names an object takes its best detected phrase."""
    actions = []
    for params, key, val in _label_params(score):
        if not isinstance(val, str) or not val or val in det_map:
            continue
        found = [p for p in same_object_phrases(val) if p in det_map]
        if found:
            params[key] = max(found, key=lambda p: float(getattr(det_map[p], 'confidence', 0.0)))
            actions.append({'event': 'object_snap', 'from': val, 'to': params[key], 'key': key})
    return actions


def track_strongest_phrase(score: dict, det_map: dict) -> list[dict]:
    """In place: a plan label that ``bind_across_views`` placed with its object's other phrases becomes the phrase
    SAM3 scored highest there. They all sit at one place, so the target does not move. A phrase SAM3 did not detect
    (it carries the floor score) is never seen again in a fresh frame, so the scene diff and re-binds would hold
    the object at its first position however far it moves."""
    actions = []
    for params, key, val in _label_params(score):
        group = getattr(det_map.get(val), 'pick_group', None) if isinstance(val, str) else None
        if not group:
            continue
        best = max((p for p in group if p in det_map), key=lambda p: float(det_map[p].confidence))
        if float(det_map[best].confidence) > float(det_map[val].confidence):
            params[key] = best
            actions.append({'event': 'track_strongest_phrase', 'from': val, 'to': best, 'key': key})
    return actions


def _run_label_validator(cfg: FairConfig, score: dict, det_map: dict, instruction: str, rgb,
                         pick_hint: str, place_hint: str, trial_meta: Optional[dict] = None,
                         slot_key: str = 'validator_actions', picks: Optional[list] = None,
                         places: Optional[list] = None, pick_counts: Optional[list] = None) -> dict:
    """
    Snap BT labels that are not in ``det_map`` to the nearest detected label (Levenshtein cutoff 0.4);
    any label left over triggers ONE replan with a STRICT preamble listing the allowed labels. Records the
    snap / retry events in ``trial_meta[slot_key]``.
    """
    if not isinstance(score, dict) or not det_map:
        return score
    allowed = set(det_map.keys()) | {'none'}
    object_snap = bool(getattr(cfg, 'validator_object_snap', False))
    pre_actions = snap_to_object_phrase(score, det_map) if object_snap else []

    def _retry_fn(suffix: str) -> Optional[dict]:
        return _plan(cfg, suffix + "\n\n" + instruction, det_map, rgb, pick_hint, place_hint,
                     picks=picks, places=places, pick_counts=pick_counts)

    repaired, actions = validate_and_repair_bt(
        score, allowed, retry_fn=_retry_fn, log=log.info if cfg.verbose else (lambda _msg: None),
        require_related=object_snap)
    actions = pre_actions + actions + (track_strongest_phrase(repaired, det_map) if isinstance(repaired, dict) else [])
    if trial_meta is not None and actions:
        trial_meta.setdefault(slot_key, []).extend(actions)
    return repaired
