"""Feasibility: count yields per denominator and simulate the generated
sample needed for a target number of strict events.

The earlier plan targeted 25 strict triggers using the *joint-event* yield.
Those are different denominators: the confirmatory run produced 4 joint
capture events but only 1 strict single-clause capture trigger from 28
generated episodes. This module keeps the denominators separate and
bootstraps over families so family-level variability is in the estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class FamilyYield:
    family: str
    generated: int
    usable: int
    joint_events: int
    strict_captures: int
    two_clause_captures: int
    committed_runs: int
    releases: int


def counts(fams: List[FamilyYield]) -> Dict[str, int]:
    keys = ["generated", "usable", "joint_events", "strict_captures", "two_clause_captures", "committed_runs", "releases"]
    return {k: int(sum(getattr(f, k) for f in fams)) for k in keys}


def episodes_needed(fams: List[FamilyYield], target: int, key: str = "strict_captures",
                    n_boot: int = 2000, seed: int = 0, quantile: float = 0.8) -> Dict[str, float]:
    """Bootstrap families; for each resample compute per-generated-episode
    yield of `key`; report the generated-episode count needed so that the
    target is reached in `quantile` of resamples."""
    rng = np.random.default_rng(seed)
    per_gen = []
    for _ in range(n_boot):
        s = rng.choice(len(fams), size=len(fams), replace=True)
        g = sum(fams[i].generated for i in s); y = sum(getattr(fams[i], key) for i in s)
        per_gen.append(y / g if g else 0.0)
    per_gen = np.array(per_gen)
    point = counts(fams)[key] / max(counts(fams)["generated"], 1)
    lo = np.quantile(per_gen, 1 - quantile)
    return {"yield_point": float(point),
            "yield_q_low": float(lo),
            "episodes_needed_point": float(target / point) if point > 0 else float("inf"),
            "episodes_needed_conservative": float(target / lo) if lo > 0 else float("inf"),
            "n_families": len(fams)}
