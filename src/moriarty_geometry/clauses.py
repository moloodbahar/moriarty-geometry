"""Clause spans with an exact-reconstruction guarantee.

The original MORIARTY splitter (goal_distribution.split_clauses) returned
clause *strings* and rebuilt the step with " ".join(...). Splitting at
",\\s+(?=and |but |...)" consumes the comma, so the rebuilt "present"
condition differed from the measured text in 5 of 11 pilot trigger
entries (2 of 5 validated). Here a clause is a character span into the
original step. Every edit is performed on spans, and every edit asserts
that the untouched text is byte-identical to the original.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
CONJ_SPLIT = re.compile(r",(\s+)(?=and |but |while |as |so )")
LONG_SENTENCE = 140  # same rule as the original splitter


@dataclass(frozen=True)
class Span:
    start: int  # inclusive, char offset into the step
    end: int    # exclusive

    def text(self, step: str) -> str:
        return step[self.start:self.end]


def split_spans(step: str) -> List[Span]:
    """Sentences first; long sentences additionally split before discourse
    connectives. The split keeps the comma with the preceding clause and the
    whitespace with the following one, so concatenating span texts in order
    reproduces the step exactly (including inter-clause whitespace, which is
    assigned to the following span)."""
    spans: List[Span] = []
    pos = 0
    text = step
    # sentence boundaries: keep the separating whitespace attached to the
    # *next* sentence so nothing is lost
    sent_starts = [0] + [m.start() for m in SENT_BOUNDARY.finditer(text)]
    sent_bounds = [(sent_starts[i], sent_starts[i + 1] if i + 1 < len(sent_starts) else len(text))
                   for i in range(len(sent_starts))]
    for s, e in sent_bounds:
        seg = text[s:e]
        core = seg.strip()
        if len(core) > LONG_SENTENCE:
            # split inside the segment at ", (?=and |...)" — comma stays left
            cut_points = [s]
            for m in CONJ_SPLIT.finditer(seg):
                cut_points.append(s + m.start() + 1)  # position right after the comma
            cut_points.append(e)
            for i in range(len(cut_points) - 1):
                a, b = cut_points[i], cut_points[i + 1]
                if text[a:b].strip():
                    spans.append(Span(a, b))
        elif core:
            spans.append(Span(s, e))
    # contiguity / coverage check
    assert spans and spans[0].start == 0 and spans[-1].end == len(text), "spans must cover the step"
    for a, b in zip(spans, spans[1:]):
        assert a.end == b.start, "spans must be contiguous"
    return spans


def clause_texts(step: str) -> List[str]:
    """Trimmed clause strings, for display and for matching against the
    original trigger records. Not for reconstruction."""
    return [sp.text(step).strip() for sp in split_spans(step)]


def reconstruct(step: str, spans: Sequence[Span]) -> str:
    return "".join(sp.text(step) for sp in spans)


def edit_step(step: str, spans: Sequence[Span], index: int, replacement: str | None) -> str:
    """Return the step with clause `index` replaced by `replacement`
    (or deleted if None). All other characters are preserved byte-for-byte.

    Whitespace handling: the span's own leading whitespace (inherited from
    the previous boundary) is kept when replacing so the sentence rhythm is
    unchanged; on deletion the span is removed together with its leading
    whitespace, and if the deleted span was first, the following span's
    leading whitespace is stripped so the step does not start with a space.
    """
    spans = list(spans)
    assert reconstruct(step, spans) == step
    out = []
    for i, sp in enumerate(spans):
        raw = sp.text(step)
        if i != index:
            out.append(raw)
            continue
        lead = raw[: len(raw) - len(raw.lstrip())]
        if replacement is None:
            continue
        out.append(lead + replacement)
    new = "".join(out)
    if replacement is None and index == 0:
        new = new.lstrip()
    # invariant: everything outside the edited span is unchanged
    before = "".join(sp.text(step) for sp in spans[:index])
    after = "".join(sp.text(step) for sp in spans[index + 1:])
    if replacement is None:
        assert new.endswith(after.lstrip() if index == 0 else after), "suffix altered"
    else:
        assert new.startswith(before) and new.endswith(after), "context altered by edit"
    return new


def prefix_incremental(step: str, spans: Sequence[Span], upto: int) -> str:
    """Step text consisting of clauses [0, upto) — used for the incremental
    addition analysis. upto == len(spans) gives the full step."""
    return reconstruct(step, list(spans)[:upto]).rstrip()


def find_clause(step: str, spans: Sequence[Span], needle: str) -> int:
    """Locate a clause by text, tolerating the truncated strings stored in
    older trigger files. Exact match first, then prefix match on ≥ 40 chars."""
    texts = [sp.text(step).strip() for sp in spans]
    if needle in texts:
        return texts.index(needle)
    key = needle.strip()[:40]
    hits = [i for i, t in enumerate(texts) if t.startswith(key) or key in t]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"clause not uniquely found: {needle[:60]!r} -> {hits}")
