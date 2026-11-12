"""Synthetic recordings with a known answer (pipeline step 0).

Generates the same structure scripts/03_record_activations.py writes, from a
generative model in which the experimenter decides whether a
position-invariant "selected" code exists:

    read-out x = common
               + sum_k  w[goal at k] * B_k r_goal      position-bound content
               + s * M r_dominant                       selected code (planted only)
               + noise

B_k are fixed position-specific orthogonal maps, M a fixed map shared by all
families, r the family-centred reference vectors, w the goal distribution.

* planted (s > 0): the clause-present prompt carries the target's content in
  a position-invariant form -> the cross-position selection effect is > 0.
* null (s = 0): the clause still changes behaviour and the position-bound
  code, but there is no selected code -> the cross-position effect must not
  be detected as positive.

Layer 0 never carries the selected code, so layer selection is exercised too.
"""
from __future__ import annotations

import pathlib
from typing import Dict

import numpy as np

from .selection import save_recording

ORDERS = [tuple((i - r) % 4 for i in range(4)) for r in range(4)]
MAPPINGS = [("C", "A", "D", "B"), ("B", "D", "A", "C"), ("D", "C", "B", "A"), ("A", "B", "C", "D")]
CELLS = [(o, m) for o in ORDERS for m in MAPPINGS]


def _orth(rng, d):
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q


def simulate(n_families: int = 12, d: int = 96, strength: float = 1.0, noise: float = 0.6,
             seed: int = 0, n_layers: int = 3, inert_peak: float = 0.80) -> Dict[str, object]:
    """inert_peak = 0.82 makes the inert-edit condition an independent draw of
    exactly the present condition: an exact null for calibrating the test."""
    rng = np.random.default_rng(seed)
    B = [_orth(rng, d) for _ in range(4)]
    M = _orth(rng, d)
    common = rng.standard_normal(d) * 2.0
    value_axis = rng.standard_normal(d)
    recs: Dict[str, object] = {"events": {}, "traj": [], "refs": {}}

    def record(p, family, episode, kind="traj", target=-1, selected=True):
        p = np.asarray(p, float)
        dom = int(np.argmax(p))
        dist, order, hx, hc = [], [], [], []
        for o, _m in CELLS:
            dist.append(rng.dirichlet(p * 60 + 1e-3))
            order.append(o)
        # read-outs before the mapping are identical across the 4 mappings of an order
        per_order = {}
        for o in ORDERS:
            layers_x, layers_c = [], []
            for li in range(n_layers):
                x = common.copy()
                for k in range(4):
                    x += p[o[k]] * (B[k] @ Rc[o[k]])
                if li > 0 and selected:
                    x += strength * p[dom] * (M @ Rc[dom])
                layers_x.append(x + noise * rng.standard_normal(d))
                layers_c.append([common + p[o[k]] * 2.0 * value_axis + B[k] @ Rc[o[k]]
                                 + noise * rng.standard_normal(d) for k in range(4)])
            per_order[o] = (np.array(layers_x), np.array(layers_c).transpose(1, 0, 2))
        for o, _m in CELLS:
            hx.append(per_order[o][0]); hc.append(per_order[o][1])
        hx = np.array(hx)
        return {"dist": np.array(dist), "order": np.array(order), "layers": np.arange(n_layers),
                "family": family, "episode": episode, "kind": kind, "target": target, "true": 0,
                "h_cand": np.array(hc), "h": {"post_list": hx, "after_candidates": hx.copy(), "answer": hx.copy()}}

    peak = lambda g, a=0.82: np.array([a if i == g else (1 - a) / 3 for i in range(4)])
    flat = np.array([0.25, 0.25, 0.25, 0.25])
    for f in range(n_families):
        family = f"s{f:02d}_sim"
        R = rng.standard_normal((4, d)) + rng.standard_normal(d)        # shared family component
        Rc = R - R.mean(0, keepdims=True)
        refs = np.stack([R + 0.05 * li for li in range(n_layers)], 1)   # (4, L, d)
        for g in range(1, 5):
            episode = f"{family}_g{g}"
            recs["refs"][episode] = refs.astype(np.float32)
            wrong = 1 + (g % 3)
            captured = g % 2 == 0
            traj_p = [flat, peak(wrong, 0.5) if captured else flat, peak(wrong) if captured else peak(0, 0.5),
                      peak(wrong) if captured else peak(0, 0.6), peak(wrong, 0.7) if captured else peak(0),
                      peak(0, 0.6), peak(0, 0.95)]
            for t, p in enumerate(traj_p):
                r = record(p, family, episode); r["t"] = t
                recs["traj"].append(r)
            if g <= 2:                      # two events per family: one capture, one release
                kind = "capture" if captured else "release"
                target = wrong if captured else 0
                eid = f"{episode}_s{2 if captured else 5}_{kind}"
                conds = {("at", "present"): record(peak(target), family, episode, kind, target),
                         ("at", "inert_neutral"): record(peak(target, inert_peak), family, episode, kind, target),
                         ("at", "cue"): record(peak(target, 0.97), family, episode, kind, target),
                         # control task: same position-bound code, no selection demanded
                         ("at", "ctl_present"): record(peak(target), family, episode, kind, target, selected=False),
                         ("at", "ctl_neutral_primary"): record(peak(target, 0.34), family, episode, kind, target, selected=False),
                         ("at", "neutral_primary"): record(peak(target, 0.34), family, episode, kind, target),
                         ("at", "neutral_secondary"): record(peak(target, 0.36), family, episode, kind, target),
                         ("pre", "present"): record(flat, family, episode, kind, target)}
                recs["events"][eid] = {"family": family, "episode": episode, "kind": kind, "target": target,
                                       "true": 0, "conds": conds}
    return recs


def write(recs: Dict[str, object], directory: str, fmt: str = "decoupled") -> None:
    """Dump simulated recordings in the on-disk format, for end-to-end tests
    of scripts/04_analyse.py."""
    d = pathlib.Path(directory); d.mkdir(parents=True, exist_ok=True)
    goals = [f"goal {i}" for i in range(4)]
    for ep, v in recs["refs"].items():
        np.savez_compressed(d / f"ref__{ep}.npz", episode=ep, goals=np.array(goals), vecs=v)

    def dump(name, r):
        save_recording(d / name, goals=goals, dist=r["dist"], order=r["order"], mapping=[m for _o, m in CELLS],
                       hidden=r["h"], h_cand=r["h_cand"], layers=r["layers"], family=r["family"],
                       episode=r["episode"], kind=r["kind"], target=r["target"], true=r["true"])
    for r in recs["traj"]:
        dump(f"{r['episode']}__t{r['t']}__traj__{fmt}.npz", r)
    for eid, ev in recs["events"].items():
        for (label, cond), r in ev["conds"].items():
            dump(f"{eid}__{label}__{cond}__{fmt}.npz", r)
