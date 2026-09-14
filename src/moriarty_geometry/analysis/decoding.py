"""Dominance probe: 'is this candidate currently dominant?'

A shared binary probe over candidate-final-token activations gives a label
that is consistent across families whose goals mean different things. The
split is by goal family (never by episode), and by presentation position
where the design allows it, so the probe cannot learn family identity or
position. Balanced accuracy is the metric because dominance is 1-of-4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


@dataclass
class Sample:
    x: np.ndarray          # (d,) activation of one candidate's final token, one layer
    y: int                 # 1 if this candidate is the dominant goal at this prefix
    family: str
    episode: str
    t: int
    position: int          # list position of the candidate
    letter: str            # answer letter assigned to the candidate
    goal_is_true: bool
    regime: str            # "pre_capture" | "committed_wrong" | "post_release" | "other"


def family_folds(samples: Sequence[Sample], n_folds: int = 5, seed: int = 0) -> List[Tuple[List[int], List[int]]]:
    fams = sorted({s.family for s in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(fams)
    groups = [fams[i::n_folds] for i in range(n_folds)]
    folds = []
    for g in groups:
        test = [i for i, s in enumerate(samples) if s.family in g]
        train = [i for i, s in enumerate(samples) if s.family not in g]
        folds.append((train, test))
    return folds


def fit_probe(X: np.ndarray, y: np.ndarray, C: float = 1.0):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
    return sc, clf


def cross_family_accuracy(samples: Sequence[Sample], n_folds: int = 5, seed: int = 0, C: float = 1.0) -> Dict[str, float]:
    X = np.stack([s.x for s in samples]); y = np.array([s.y for s in samples])
    accs = []
    for train, test in family_folds(samples, n_folds, seed):
        sc, clf = fit_probe(X[train], y[train], C)
        accs.append(balanced_accuracy_score(y[test], clf.predict(sc.transform(X[test]))))
    return {"balanced_accuracy_mean": float(np.mean(accs)), "balanced_accuracy_sd": float(np.std(accs)), "n_folds": n_folds}


def select_layer(samples_by_layer: Dict[int, Sequence[Sample]], **kw) -> Tuple[int, Dict[int, float]]:
    """Layer selection on the TRAINING split only: return the layer with the
    highest cross-family balanced accuracy and the full curve."""
    curve = {l: cross_family_accuracy(s, **kw)["balanced_accuracy_mean"] for l, s in samples_by_layer.items()}
    best = max(curve, key=curve.get)
    return best, curve


def wrong_goal_evidence(sc, clf, samples: Sequence[Sample]) -> np.ndarray:
    """Probe log-odds that each (wrong) candidate is dominant — the quantity
    whose time-course H2 tests."""
    X = np.stack([s.x for s in samples])
    return clf.decision_function(sc.transform(X))


def regime_contrast(evidence: np.ndarray, samples: Sequence[Sample], a: str, b: str) -> Dict[str, float]:
    """Mean evidence in regime a minus regime b, restricted to wrong
    candidates, with episode-level aggregation so episodes are the unit."""
    per_ep: Dict[str, Dict[str, List[float]]] = {}
    for e, s in zip(evidence, samples):
        if s.goal_is_true:
            continue
        per_ep.setdefault(s.episode, {}).setdefault(s.regime, []).append(float(e))
    diffs = [np.mean(r[a]) - np.mean(r[b]) for r in per_ep.values() if a in r and b in r]
    if not diffs:
        return {"n_episodes": 0}
    d = np.array(diffs)
    return {"n_episodes": len(d), "mean_diff": float(d.mean()),
            "ci95": (float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                     float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 else None}
