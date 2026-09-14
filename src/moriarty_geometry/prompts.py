"""Two probe prompt formats.

NATIVE reproduces the MORIARTY probe (paper App. E.8): goals are listed as
"A. goal" ... "D. goal" and cycled through the four letter positions. In this
format a goal's list position and its answer letter move together, so the
format cannot separate position coding from letter coding.

DECOUPLED lists the goals as a numbered list and gives the answer mapping
separately ("first -> C; second -> A; ..."). List order and answer mapping
are varied independently. It also exposes a marker AFTER_CANDIDATES: the
character offset (in the user message) right after the last candidate and
before the mapping. A hidden state read there cannot depend on the mapping,
which the model has not yet seen.

Both builders return a PromptSpec with the system text, user text, the
letter->goal map for reading the answer distribution, and character markers.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

LETTERS = ["A", "B", "C", "D"]
ORDINALS = ["first", "second", "third", "fourth"]

SYSTEM_NATIVE = (
    "You are given the opening of a story and four candidate hidden goals the "
    "story's author might be pursuing. Answer which goal is most likely with a "
    "SINGLE LETTER (A, B, C, or D) and nothing else."
)
SYSTEM_DECOUPLED = (
    "You are given the opening of a story and four numbered candidate hidden "
    "goals the story's author might be pursuing. After the list you will be "
    "given a mapping from list position to answer letter. Answer which goal is "
    "most likely with the SINGLE LETTER assigned to it, and nothing else."
)

T0_TEXT = "(the story has not started yet)"


@dataclass
class PromptSpec:
    system: str
    user: str
    letter_to_goal: Dict[str, str]          # answer letter -> goal text
    order: Tuple[int, ...]                  # goal index (in canonical list) at each list position
    mapping: Tuple[str, ...]                # answer letter for each list position
    markers: Dict[str, int] = field(default_factory=dict)  # name -> char offset in `user`
    fmt: str = "native"

    @property
    def answer_marker(self) -> int:
        return len(self.user)


def story_block(steps: Sequence[str]) -> str:
    if not steps:
        return T0_TEXT
    return "\n".join(f"[step {i + 1}] {s}" for i, s in enumerate(steps))


def rotations_native(n: int = 4) -> List[Tuple[int, ...]]:
    """The four cyclic rotations used by MORIARTY: goal i sits at position
    (i + r) mod 4 for r = 0..3."""
    return [tuple((i - r) % n for i in range(n)) for r in range(n)]


def build_native(steps: Sequence[str], goals: Sequence[str], rotation: int,
                 chars_block: Optional[str] = None) -> PromptSpec:
    """goals: canonical list [true, d1, d2, d3] or any fixed order — the
    caller keeps the canonical index. rotation r places goal i at letter
    position (i + r) % 4, matching the original probe."""
    order = rotations_native()[rotation]          # order[pos] = goal index at that position
    lines = [f"STORY OPENING:\n{story_block(steps)}", ""]
    if chars_block:
        lines += [f"CHARACTERS:\n{chars_block}", ""]
    lines.append("CANDIDATE HIDDEN GOALS:")
    letter_to_goal = {}
    for pos, gi in enumerate(order):
        lines.append(f"{LETTERS[pos]}. {goals[gi]}")
        letter_to_goal[LETTERS[pos]] = goals[gi]
    user = "\n".join(lines)
    markers = {"after_candidates": len(user)}
    user += "\n\nWhich is most likely? Answer with a single letter."
    return PromptSpec(SYSTEM_NATIVE, user, letter_to_goal, order, tuple(LETTERS), markers, "native")


def all_orders(n: int = 4) -> List[Tuple[int, ...]]:
    return list(itertools.permutations(range(n)))


def latin_orders() -> List[Tuple[int, ...]]:
    """Four orders in which every goal appears once in every position
    (a Latin square) — the default balanced set for the decoupled format."""
    return rotations_native()


def latin_mappings() -> List[Tuple[str, ...]]:
    """Four mappings in which every list position receives every letter once,
    chosen to be *derangements* of the identity so that position != letter
    in every mapping (mapping[pos] is the letter for list position pos)."""
    return [("C", "A", "D", "B"), ("B", "D", "A", "C"), ("D", "C", "B", "A"), ("A", "B", "C", "D")]


def build_decoupled(steps: Sequence[str], goals: Sequence[str], order: Sequence[int],
                    mapping: Sequence[str], chars_block: Optional[str] = None) -> PromptSpec:
    """order[pos] = canonical goal index shown at list position pos (1-based
    in the prompt); mapping[pos] = answer letter assigned to that position."""
    assert sorted(order) == [0, 1, 2, 3] and sorted(mapping) == LETTERS
    lines = [f"STORY OPENING:\n{story_block(steps)}", ""]
    if chars_block:
        lines += [f"CHARACTERS:\n{chars_block}", ""]
    lines.append("CANDIDATE HIDDEN GOALS:")
    for pos, gi in enumerate(order):
        lines.append(f"{pos + 1}. {goals[gi]}")
    user = "\n".join(lines)
    markers = {"after_candidates": len(user)}
    map_str = "; ".join(f"{ORDINALS[pos]} -> {mapping[pos]}" for pos in range(4))
    user += f"\n\nANSWER MAPPING: {map_str}."
    markers["after_mapping"] = len(user)
    user += "\n\nWhich is most likely? Answer with the single letter assigned to it."
    letter_to_goal = {mapping[pos]: goals[order[pos]] for pos in range(4)}
    return PromptSpec(SYSTEM_DECOUPLED, user, letter_to_goal, tuple(order), tuple(mapping), markers, "decoupled")


def decoupled_design(orders: Optional[List[Tuple[int, ...]]] = None,
                     mappings: Optional[List[Tuple[str, ...]]] = None) -> List[Tuple[Tuple[int, ...], Tuple[str, ...]]]:
    """Full factorial of the balanced orders x balanced mappings (16 cells by
    default). Each goal then appears in every position and under every
    letter, and position and letter are crossed rather than confounded."""
    orders = orders or latin_orders()
    mappings = mappings or latin_mappings()
    return [(o, m) for o in orders for m in mappings]


def chars_block_level2(cast: Sequence[dict], visible: Sequence[str]) -> str:
    """Level-2 information block: visible cores shown, others marked unknown."""
    out = []
    for c in cast:
        core = c["moral_core"] if c["name"] in visible else "(unknown)"
        out.append(f"- {c['name']}: moral core = {core}")
    return "\n".join(out)
