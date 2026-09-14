"""Paired activation patching.

For an event at a fixed prefix, h_P is the activation under the original
clause (present) and h_N under the neutral replacement. A subspace U is
learned from paired differences (h_P − h_N) on TRAINING families; its layer
and rank are chosen on DEVELOPMENT families; then frozen.

On held-out events the patch is
    h_N^patched = h_N + U Uᵀ (h_P − h_N)          (restore)
    h_P^patched = h_P − U Uᵀ (h_P − h_N)          (remove)
so the donor is the event's own paired difference and the only learned
object is the subspace. The supported claim is conditional reconstruction of
a clause effect — not creation of an arbitrary interpretation from a
universal "wrong-goal direction", which the original plan assumed and which
need not exist across events whose wrong goals differ in content.

Controls: random subspaces of the same rank (matched norm of the projected
difference), patches at inert-clause events, and unpatched baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Subspace:
    layer: int
    U: np.ndarray            # (d, k), orthonormal columns
    explained: np.ndarray    # variance ratio per component

    @property
    def rank(self) -> int:
        return self.U.shape[1]

    def project(self, v: np.ndarray) -> np.ndarray:
        return self.U @ (self.U.T @ v)


def learn_subspace(diffs: np.ndarray, rank: int, layer: int) -> Subspace:
    """diffs: (n_events, d) paired differences h_P − h_N at one layer.
    PCA (uncentred: the differences themselves carry the signal)."""
    U, S, _ = np.linalg.svd(diffs, full_matrices=False)
    comps = np.linalg.svd(diffs, full_matrices=False)[2][:rank].T   # (d, k)
    var = S ** 2 / np.sum(S ** 2)
    return Subspace(layer, comps, var[:rank])


def random_subspace(d: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, rank))
    Q, _ = np.linalg.qr(A)
    return Q


def patch_restore(h_N: np.ndarray, h_P: np.ndarray, U: np.ndarray) -> np.ndarray:
    return h_N + U @ (U.T @ (h_P - h_N))


def patch_remove(h_N: np.ndarray, h_P: np.ndarray, U: np.ndarray) -> np.ndarray:
    return h_P - U @ (U.T @ (h_P - h_N))


def matched_random_patch(h_N: np.ndarray, h_P: np.ndarray, U: np.ndarray, seed: int) -> np.ndarray:
    """Random subspace of the same rank; the injected vector is rescaled to
    the norm of the learned-subspace injection so the two differ only in
    direction."""
    d, k = U.shape
    R = random_subspace(d, k, seed)
    inj = U @ (U.T @ (h_P - h_N))
    rnd = R @ (R.T @ (h_P - h_N))
    scale = np.linalg.norm(inj) / (np.linalg.norm(rnd) + 1e-12)
    return h_N + scale * rnd


@dataclass
class PatchOutcome:
    event_id: str
    condition: str          # "baseline_N" | "baseline_P" | "restore" | "remove" | "random_k" | "inert"
    p_target: float         # probability on the event's target goal after the patch
    dominant: str
    coverage: float


def select_rank_and_layer(diffs_by_layer: Dict[int, np.ndarray], dev_eval: Callable[[Subspace], float],
                          ranks: Sequence[int] = (1, 2, 4, 8)) -> Subspace:
    """Choose (layer, rank) maximising the development-set restore effect.
    dev_eval(Subspace) -> mean Δp_target on DEVELOPMENT events. Called once;
    the returned subspace is then frozen for the held-out test."""
    best, best_val = None, -np.inf
    for layer, diffs in diffs_by_layer.items():
        for r in ranks:
            if r > min(diffs.shape):
                continue
            S = learn_subspace(diffs, r, layer)
            v = dev_eval(S)
            if v > best_val:
                best, best_val = S, v
    return best


def summarise(outcomes: Sequence[PatchOutcome], target_condition: str, control_condition: str,
              min_coverage: float = 0.5) -> Dict[str, float]:
    """Difference in p_target between a patch condition and its control,
    episode-paired, excluding degenerate calls (coverage below threshold)."""
    by_ev: Dict[str, Dict[str, PatchOutcome]] = {}
    for o in outcomes:
        by_ev.setdefault(o.event_id, {})[o.condition] = o
    diffs = []
    dropped = 0
    for ev, d in by_ev.items():
        if target_condition in d and control_condition in d:
            a, b = d[target_condition], d[control_condition]
            if a.coverage < min_coverage or b.coverage < min_coverage:
                dropped += 1
                continue
            diffs.append(a.p_target - b.p_target)
    if not diffs:
        return {"n": 0, "dropped_low_coverage": dropped}
    x = np.array(diffs)
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return {"n": len(x), "mean_delta_p": float(x.mean()), "ci95": (float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)),
            "dropped_low_coverage": dropped}
