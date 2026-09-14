"""Trajectory quantities and threshold events, carried over unchanged from
MORIARTY (Section 3.2). Thresholds are module constants so the
preregistration hash covers them."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

DH_UNCERTAINTY = 0.15
RESOLUTION_GAIN = 0.30
RESOLUTION_AGREE = 3          # of 4 rotations
CW_Q_MAX, CW_W_MIN, CW_H_MAX = 0.20, 0.70, 0.50


def entropy_norm(p: Dict[str, float]) -> float:
    n = len(p)
    return -sum(v * math.log2(v) for v in p.values() if v > 0) / math.log2(n)


def jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    m = {k: 0.5 * (p[k] + q[k]) for k in p}
    def kl(a, b):
        return sum(a[k] * math.log2(a[k] / b[k]) for k in a if a[k] > 0)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@dataclass
class Point:
    t: int
    p: Dict[str, float]
    q: float
    W: float
    H: float
    dominant: str
    dominant_is_true: bool
    argmax_agreement: int      # how many rotations' argmax equals the averaged dominant goal
    coverage_min: float
    dq: Optional[float] = None
    dH: Optional[float] = None
    dW: Optional[float] = None
    jsd_prev: Optional[float] = None


def make_point(t: int, p: Dict[str, float], true_goal: str, argmaxes: Sequence[str],
               coverages: Sequence[float], prev: Optional[Point] = None) -> Point:
    dom = max(p, key=p.get)
    pt = Point(t, p, p[true_goal], max(v for g, v in p.items() if g != true_goal),
               entropy_norm(p), dom, dom == true_goal,
               sum(1 for a in argmaxes if a == dom), min(coverages))
    if prev is not None:
        pt.dq, pt.dH, pt.dW = pt.q - prev.q, pt.H - prev.H, pt.W - prev.W
        pt.jsd_prev = jsd(pt.p, prev.p)
    return pt


def events(points: Sequence[Point]) -> Dict[str, object]:
    """Detect the five MORIARTY events on a trajectory (t = 0..T)."""
    ev: Dict[str, object] = {"uncertainty_creation": [], "wrong_entry": None,
                             "wrong_collapse": None, "committed_wrong": [], "resolution": None}
    for a, b in zip(points, points[1:]):
        if b.dH is not None and b.dH >= DH_UNCERTAINTY:
            ev["uncertainty_creation"].append(b.t)
        if a.dominant_is_true and not b.dominant_is_true and b.argmax_agreement >= RESOLUTION_AGREE \
                and ev["wrong_entry"] is None:
            ev["wrong_entry"] = b.t
    # wrong collapse: strongest reliable point with dW>0, dq<0, dH<0, wrong dominant
    cands = [p for p in points[1:] if p.dW is not None and p.dW > 0 and p.dq < 0 and p.dH < 0
             and not p.dominant_is_true and p.argmax_agreement >= RESOLUTION_AGREE]
    if cands:
        ev["wrong_collapse"] = max(cands, key=lambda p: p.dW).t
    # committed-wrong runs
    run: List[int] = []
    for p in points:
        if p.q <= CW_Q_MAX and p.W >= CW_W_MIN and p.H <= CW_H_MAX:
            run.append(p.t)
        else:
            if len(run) >= 1:
                ev["committed_wrong"].append((run[0], run[-1]))
            run = []
    if run:
        ev["committed_wrong"].append((run[0], run[-1]))
    # resolution
    for p in points[1:]:
        if p.dq is not None and p.dq >= RESOLUTION_GAIN and p.dH < 0 and p.dominant_is_true \
                and p.argmax_agreement >= RESOLUTION_AGREE:
            ev["resolution"] = p.t
            break
    return ev
