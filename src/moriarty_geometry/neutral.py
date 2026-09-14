"""Per-event neutral replacements.

The original control drew from a bank of eight generic indoor sentences
chosen by character length. Those sentences are world-incongruous in most
episodes (a clock on a wall, a phone buzzing, in a mountain-rescue post or a
fishing village) — an incongruity confound on top of the manipulation.

Here every event has its own replacement(s), written before measurement,
validated by the checks below, and frozen by hash. Each record carries:
  event_id, episode_id, step, clause_index, trigger_clause (exact span text),
  neutral_primary, neutral_secondary, inert_index (index of an inert clause
  in the same step), inert_neutral, notes.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from .clauses import split_spans, edit_step

LENGTH_TOL = 0.20


@dataclass
class NeutralRecord:
    event_id: str
    episode_id: str
    step: int
    clause_index: int
    trigger_clause: str
    target_goal: str                 # the goal whose probability the trigger moves
    trigger_type: str                # "capture" or "release"
    neutral_primary: str
    neutral_secondary: str
    inert_index: int
    inert_neutral: str
    character: str
    notes: str = ""
    status: str = "DRAFT"            # DRAFT -> JUDGED -> FROZEN


def _len_ok(a: str, b: str) -> bool:
    return abs(len(b) - len(a)) <= LENGTH_TOL * max(len(a), 1)


def validate_record(rec: NeutralRecord, step_text: str, goals: List[str],
                    cast_names: Optional[List[str]] = None) -> List[str]:
    """Return a list of problems (empty = passes the mechanical checks).
    The judge checks (grammaticality, no new goal-relevant information) are
    recorded separately; this function covers what can be checked
    deterministically."""
    problems = []
    spans = split_spans(step_text)
    if rec.clause_index >= len(spans):
        return [f"clause_index {rec.clause_index} out of range ({len(spans)} clauses)"]
    orig = spans[rec.clause_index].text(step_text).strip()
    if orig != rec.trigger_clause.strip():
        problems.append("trigger_clause does not match the span text exactly")
    for name, rep in (("neutral_primary", rec.neutral_primary), ("neutral_secondary", rec.neutral_secondary)):
        if not _len_ok(orig, rep):
            problems.append(f"{name} length outside ±{int(LENGTH_TOL*100)}% ({len(rep)} vs {len(orig)})")
        if rec.character and rec.character not in rep:
            problems.append(f"{name} does not mention the character '{rec.character}'")
        names = {n.lower() for n in (cast_names or [])}
        for g in goals:
            for w in _content_words(g):
                if w in names:
                    continue
                if w in rep.lower() and w not in orig.lower():
                    problems.append(f"{name} introduces goal-related word '{w}'")
        # edit must leave context untouched (asserted inside edit_step)
        edit_step(step_text, spans, rec.clause_index, rep)
    if rec.inert_index == rec.clause_index or rec.inert_index >= len(spans):
        problems.append("inert_index must point at a different clause of the same step")
    else:
        inert = spans[rec.inert_index].text(step_text).strip()
        if not _len_ok(inert, rec.inert_neutral):
            problems.append("inert_neutral length outside tolerance")
    return problems


_STOP = set("the a an to of in on at for with and or but that this his her their its be is was were get "
            "make lead let have has into from by as it he she they them him one own new every everyone "
            "everything before after while which would could should there where about again".split())


def _content_words(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 4 and w not in _STOP]


def freeze_hash(records: List[NeutralRecord]) -> str:
    payload = json.dumps([asdict(r) for r in records], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_registry(path: str) -> List[NeutralRecord]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return [NeutralRecord(**r) for r in d["records"]]


def save_registry(path: str, records: List[NeutralRecord]) -> str:
    h = freeze_hash(records)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"sha256": h, "records": [asdict(r) for r in records]}, f, indent=2, ensure_ascii=False)
    return h
