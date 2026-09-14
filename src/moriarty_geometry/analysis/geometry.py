"""Representational geometry.

Two analyses with different claims:

1. cross_position_generalisation — the Li et al. (2026) analogue. Train a
   decoder to recover the *content* (which goal) of the selected candidate
   when it appeared in position i; test on prefixes where the same content
   appeared in position j != i, in held-out families. Repeat for unselected
   candidates. The prediction is that selected-content decoding generalises
   across positions more after the decisive cue than before.

2. ccgp / parallelism — Bernardi et al. (2020). For a dichotomy (e.g.
   dominant-vs-rejected, or position-1-vs-2), CCGP trains on some conditions
   and tests on held-out conditions of the same dichotomy; parallelism is the
   mean cosine between coding vectors across condition pairs. Used as the
   secondary, geometry-level analysis; kept secondary until the pilot shows
   the estimates are stable.

All functions take plain numpy arrays and condition labels so they can be
unit-tested on synthetic data.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


def _fit(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
    return sc, clf


# ------------------------------------------------------- cross-position decoding
def cross_position_generalisation(X: np.ndarray, content: np.ndarray, position: np.ndarray,
                                  family: np.ndarray, train_pos: int, test_pos: int) -> float:
    """X: (n, d) activations at a fixed read-out point; content: goal id
    (int) of the candidate whose content is to be recovered; position: list
    position where that candidate appeared; family: family id.
    Train on rows with position == train_pos, test on rows with
    position == test_pos, families disjoint. Returns balanced accuracy."""
    fams = np.unique(family)
    rng = np.random.default_rng(0)
    rng.shuffle(fams)
    test_f = set(fams[: max(1, len(fams) // 4)])
    tr = (position == train_pos) & ~np.isin(family, list(test_f))
    te = (position == test_pos) & np.isin(family, list(test_f))
    if tr.sum() < 4 or te.sum() < 2 or len(np.unique(content[tr])) < 2:
        return float("nan")
    sc, clf = _fit(X[tr], content[tr])
    return float(balanced_accuracy_score(content[te], clf.predict(sc.transform(X[te]))))


# ------------------------------------------------------------------ CCGP
def ccgp(X: np.ndarray, dichotomy: np.ndarray, condition: np.ndarray) -> float:
    """Cross-condition generalisation performance. `condition` labels the
    cells of the design (e.g. goal x position); `dichotomy` is the binary
    variable of interest. For every split of conditions into train/test
    that keeps both dichotomy values on each side, train on train-conditions
    and test on test-conditions; return the mean balanced accuracy."""
    conds = np.unique(condition)
    d_of = {c: int(np.round(dichotomy[condition == c].mean())) for c in conds}
    accs = []
    for k in range(1, len(conds) // 2 + 1):
        for test_c in itertools.combinations(conds, k):
            train_c = [c for c in conds if c not in test_c]
            if len({d_of[c] for c in test_c}) < 2 or len({d_of[c] for c in train_c}) < 2:
                continue
            tr = np.isin(condition, train_c); te = np.isin(condition, list(test_c))
            sc, clf = _fit(X[tr], dichotomy[tr])
            accs.append(balanced_accuracy_score(dichotomy[te], clf.predict(sc.transform(X[te]))))
    return float(np.mean(accs)) if accs else float("nan")


def parallelism_score(X: np.ndarray, dichotomy: np.ndarray, condition: np.ndarray,
                      pair_key: Optional[np.ndarray] = None) -> float:
    """Mean cosine similarity between coding vectors of the dichotomy
    computed in different *matched* pairs of conditions (Bernardi et al.
    2020). A coding vector for a pair (c+, c−) is mean(X|c+) − mean(X|c−).
    `pair_key` labels which conditions are matched (same value = same cell
    of the other factors, differing only in the dichotomy). If omitted, all
    c+/c− pairs are used, which mixes in between-condition differences and
    is only suitable when the other factors are absent."""
    conds = np.unique(condition)
    d_of = {c: bool(dichotomy[condition == c].mean() > 0.5) for c in conds}
    if pair_key is not None:
        k_of = {c: pair_key[condition == c][0] for c in conds}
        pairs = []
        for k in sorted(set(k_of.values())):
            cs = [c for c in conds if k_of[c] == k]
            for cp in [c for c in cs if d_of[c]]:
                for cn in [c for c in cs if not d_of[c]]:
                    pairs.append((cp, cn))
    else:
        pairs = [(cp, cn) for cp in conds if d_of[cp] for cn in conds if not d_of[cn]]
    vecs = []
    for cp, cn in pairs:
        v = X[condition == cp].mean(0) - X[condition == cn].mean(0)
        n = np.linalg.norm(v)
        if n > 0:
            vecs.append(v / n)
    if len(vecs) < 2:
        return float("nan")
    return float(np.mean([float(u @ w) for u, w in itertools.combinations(vecs, 2)]))


# ---------------------------------------------------------------- shuffle null
def shuffle_null(stat_fn, X: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                 n: int = 1000, seed: int = 0, **kw) -> Tuple[float, np.ndarray, float]:
    """Permute `labels` within `groups` (episode x step) n times and return
    (observed, null distribution, p-value one-sided upper)."""
    rng = np.random.default_rng(seed)
    obs = stat_fn(X, labels, **kw)
    null = np.empty(n)
    for i in range(n):
        perm = labels.copy()
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            perm[idx] = rng.permutation(labels[idx])
        null[i] = stat_fn(X, perm, **kw)
    p = float((np.sum(null >= obs) + 1) / (n + 1))
    return float(obs), null, p
