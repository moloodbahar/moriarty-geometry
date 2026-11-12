"""Selection index: does the read-out carry the *content* of the selected
goal in a form that does not depend on where that goal sat in the list?

This is the primary test of the Li et al. (2026, iScience, PMID 41676500)
analogue. It replaces the first version of scripts/04_analyse.py, which had
two defects (DECISIONS.md §5):

* it gave all four candidates of a prompt the same activation vector with
  different `dominant` labels, so nothing could be decoded;
* it used the canonical goal index as "content" across families, but
  goals[0] is the true goal in every episode, so that index means
  "true-goal slot", not content.

Content is therefore defined in a space shared across families: each goal
text has a context-free reference vector (probe.reference_vectors). A ridge
map is fitted, on TRAINING families only, from the read-out activation to
the (family-centred) reference vector of the goal that is dominant at that
prefix. One map is fitted per list position of the dominant goal.

    SI(x; map, target) = cos(map(x), r_target) - mean_{g != target} cos(map(x), r_g)

    cross-position SI : map fitted on rows whose dominant goal sat at
                        position i, applied to rows whose target sits at
                        j != i (averaged over i).  <- primary
    same-position SI  : map fitted at j, applied at j.
    direct SI         : no map; cos between the read-out itself and the
                        same-layer reference vectors.

The effect for an event is SI(clause present) - SI(neutral replacement) at
the same prefix. The inert-clause edit gives the size of a pure edit effect.

Null behaviour. If content is coded only in position-bound form, a map
fitted at position i reads the goal sitting at i. In the cross-position case
that goal is a non-target, so it is excluded from the comparison set;
otherwise the clause lowering that goal's evidence would by itself produce a
positive present-minus-neutral effect. scripts/00b_synthetic_validation.py
checks that the null simulation is not detected. The hypothesis predicts a
positive effect; the test is one-sided.

Unit of analysis and test. Cells of the 16-cell design are repeated measures
of one event; events are clustered in families. The TEST is a family-level
sign-flip permutation test (family_signflip), one-sided. A family-clustered
bootstrap interval is reported for description only: under an exact null it
rejected about 10% of the time at a nominal 5% with 4 or 7 held-out families
(DECISIONS.md §6.1). With n held-out families the smallest attainable p is
2^-n, so fewer than 5 cannot reach 0.05 and 6 is the working minimum.

Controls reported with the same statistic:
  cue       positive control. Neutral step + an explicit statement of the
            target goal. If this is not detected the instrument is blind on
            this model and a null on real events means nothing.
  ctl_*     lexical control. Same prompt under an instruction that needs no
            goal inference. The clause effect there is word overlap/priming;
            the trigger effect must exceed it.
"""
from __future__ import annotations

import glob
import pathlib
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EVENT_CONDS = ("present", "neutral_primary", "neutral_secondary", "inert_neutral", "cue",
               "ctl_present", "ctl_neutral_primary")
# name -> (condition a, condition b): effect = SI(a) - SI(b)
CONTRASTS = {"present_minus_neutral_primary": ("present", "neutral_primary"),
             "present_minus_neutral_secondary": ("present", "neutral_secondary"),
             "present_minus_inert_neutral": ("present", "inert_neutral"),
             "cue_minus_neutral_primary": ("cue", "neutral_primary"),
             "ctl_present_minus_ctl_neutral": ("ctl_present", "ctl_neutral_primary")}
MIN_TEST_FAMILIES = 6


# ------------------------------------------------------------------ utilities
def family_of(episode_id: str) -> str:
    m = re.match(r"^(.*)_g\d+", episode_id)
    return m.group(1) if m else episode_id.split("_")[0]


def family_split(families: Sequence[str], seed: int = 0,
                 frac: Tuple[float, float, float] = (0.5, 0.2, 0.3)) -> Tuple[List[str], List[str], List[str]]:
    """Train / development / test split at the FAMILY level. Needs >= 3."""
    fams = sorted(set(families))
    if len(fams) < 3:
        raise ValueError(f"need at least 3 families with events for a train/dev/test split, have {len(fams)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(fams)
    n = len(fams)
    n_test = max(1, int(round(frac[2] * n)))
    n_dev = max(1, int(round(frac[1] * n)))
    n_train = n - n_dev - n_test
    if n_train < 1:
        n_train, n_dev, n_test = 1, 1, n - 2
    return fams[:n_train], fams[n_train:n_train + n_dev], fams[n_train + n_dev:]


def family_signflip(values: Sequence[float], clusters: Sequence[str], n_random: int = 20000,
                    seed: int = 0) -> Optional[float]:
    """One-sided sign-flip permutation p-value for 'mean > 0', with the
    FAMILY as the exchangeable unit (events are averaged within family).
    Exact enumeration up to 14 families. Returns None with < 2 families."""
    v = np.asarray(values, float); c = np.asarray(clusters)
    ok = ~np.isnan(v); v, c = v[ok], c[ok]
    f = np.array([v[c == k].mean() for k in np.unique(c)])
    n = len(f)
    if n < 2:
        return None
    if n <= 14:
        signs = np.array([[1.0 if (i >> k) & 1 else -1.0 for k in range(n)] for i in range(2 ** n)])
    else:
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_random, n))
        signs[0] = 1.0
    return float(np.mean((signs * f).mean(1) >= f.mean() - 1e-12))


def cluster_bootstrap(values: Sequence[float], clusters: Sequence[str], n: int = 2000,
                      seed: int = 0) -> Dict[str, object]:
    """Mean with a 90% percentile interval resampling CLUSTERS (families),
    plus the family sign-flip p-value. The interval is DESCRIPTIVE (it is
    too narrow with few families); `p_signflip` is the test."""
    v = np.asarray(values, float)
    c = np.asarray(clusters)
    ok = ~np.isnan(v)
    v, c = v[ok], c[ok]
    if len(v) == 0:
        return {"n_events": 0, "n_families": 0, "mean": None, "ci90": None, "p_signflip": None}
    by = {k: v[c == k] for k in np.unique(c)}
    keys = list(by)
    out = {"n_events": int(len(v)), "n_families": len(keys), "mean": float(v.mean()), "ci90": None,
           "p_signflip": family_signflip(v, c, seed=seed)}
    if len(keys) >= 2:
        rng = np.random.default_rng(seed)
        boots = np.empty(n)
        for b in range(n):
            pick = rng.integers(0, len(keys), len(keys))
            boots[b] = np.concatenate([by[keys[i]] for i in pick]).mean()
        out["ci90"] = (float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95)))
    return out


def unique_order_cells(order: np.ndarray) -> List[int]:
    """One cell per distinct list order. Read-outs taken BEFORE the answer
    mapping (candidate ends, after_candidates) are identical across the four
    mappings of an order; counting them four times would be duplication."""
    seen, keep = set(), []
    for c, o in enumerate(order):
        k = tuple(int(i) for i in o)
        if k not in seen:
            seen.add(k)
            keep.append(c)
    return keep


class RidgeMap:
    """Kernel ridge regression X (n, d) -> Y (n, k); n << d."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, alpha: float = 1.0):
        X = np.asarray(X, np.float64)
        self.mu = X.mean(0)
        self.Xc = X - self.mu
        K = self.Xc @ self.Xc.T
        lam = alpha * np.trace(K) / max(len(K), 1) + 1e-9
        self.A = np.linalg.solve(K + lam * np.eye(len(K)), np.asarray(Y, np.float64))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, np.float64) - self.mu) @ self.Xc.T) @ self.A


def selection_index(pred: np.ndarray, refs_c: np.ndarray, target: int, exclude: Optional[int] = None) -> float:
    """cos(pred, r_target) - mean cos(pred, r_other). refs_c: (4, d), centred.
    `exclude` drops one goal from the comparison set (see the cross-position
    case in _event_si)."""
    num = refs_c @ pred
    den = np.linalg.norm(refs_c, axis=1) * np.linalg.norm(pred) + 1e-12
    cos = num / den
    others = [g for g in range(len(cos)) if g != target and g != exclude]
    return float(cos[target] - cos[others].mean())


# ------------------------------------------------------------------------ I/O
def save_recording(path, *, goals, dist, order, mapping, hidden: Dict[str, np.ndarray], h_cand,
                   layers, family, episode, kind="traj", target=-1, true=0, coverage=None, cand_cells=None) -> None:
    """h_cand: (n_cand_cells, 4, L, d). Candidate tokens precede the answer
    mapping, so they are stored once per distinct list order; `cand_cells`
    gives the design-cell index of each stored row."""
    cand_cells = list(range(len(h_cand))) if cand_cells is None else list(cand_cells)
    np.savez_compressed(
        path, goals=np.array(goals), dist=np.asarray(dist, np.float32), order=np.asarray(order),
        mapping=np.asarray(mapping), layers=np.asarray(layers), family=family, episode=episode, kind=kind,
        target=target, true=true, h_cand=np.asarray(h_cand, np.float16), cand_cells=np.asarray(cand_cells),
        coverage=np.asarray(coverage if coverage is not None else [], np.float32),
        **{f"h_{m}": np.asarray(h, np.float16) for m, h in hidden.items()})


def _load(path) -> Dict[str, object]:
    z = np.load(path, allow_pickle=True)
    rec = {"dist": z["dist"], "order": z["order"], "layers": z["layers"], "family": str(z["family"]),
           "episode": str(z["episode"]), "kind": str(z["kind"]), "target": int(z["target"]),
           "true": int(z["true"]), "h_cand": z["h_cand"],
           "cand_cells": [int(c) for c in z["cand_cells"]] if "cand_cells" in z.files else None,
           "h": {k[2:]: z[k] for k in z.files if k.startswith("h_") and k != "h_cand"}}
    return rec


def load_recordings(directory: str, fmt: str = "decoupled") -> Dict[str, object]:
    """Read what scripts/03_record_activations.py wrote."""
    d = pathlib.Path(directory)
    recs: Dict[str, object] = {"events": {}, "traj": [], "refs": {}}
    for f in sorted(glob.glob(str(d / "ref__*.npz"))):
        z = np.load(f, allow_pickle=True)
        recs["refs"][str(z["episode"])] = z["vecs"].astype(np.float32)
    for f in sorted(glob.glob(str(d / f"*__{fmt}.npz"))):
        name = pathlib.Path(f).name
        unit, label, cond, _ = name[:-4].split("__")
        rec = _load(f)
        if cond == "traj":
            rec["t"] = int(label[1:])
            recs["traj"].append(rec)
        else:
            ev = recs["events"].setdefault(unit, {"family": rec["family"], "episode": rec["episode"],
                                                  "kind": rec["kind"], "target": rec["target"],
                                                  "true": rec["true"], "conds": {}})
            ev["conds"][(label, cond)] = rec
    return recs


# ------------------------------------------------------------------- analysis
def _refs_centred(recs, episode: str, li: int) -> np.ndarray:
    R = np.asarray(recs["refs"][episode][:, li, :], np.float64)
    return R - R.mean(0, keepdims=True)


def _training_rows(recs, families: Sequence[str], marker: str, li: int, min_dom: float):
    """Rows (one per distinct list order) from training families where one
    goal is dominant: X = read-out, Y = centred reference of the dominant
    goal, pos = list position of the dominant goal."""
    fams = set(families)
    sources = [r for r in recs["traj"] if r["family"] in fams]
    for ev in recs["events"].values():
        if ev["family"] in fams:
            sources += [r for (label, cond), r in ev["conds"].items() if cond == "present"]
    X, Y, pos = [], [], []
    for r in sources:
        pbar = r["dist"].mean(0)
        dom = int(np.argmax(pbar))
        if pbar[dom] < min_dom or r["episode"] not in recs["refs"]:
            continue
        Rc = _refs_centred(recs, r["episode"], li)
        for c in unique_order_cells(r["order"]):
            X.append(r["h"][marker][c, li])
            Y.append(Rc[dom])
            pos.append(int(np.where(r["order"][c] == dom)[0][0]))
    if not X:
        return np.zeros((0, 1)), np.zeros((0, 1)), np.zeros(0, int)
    return np.asarray(X, np.float64), np.asarray(Y, np.float64), np.asarray(pos)


def _event_si(recs, ev, cond: str, marker: str, li: int, maps: Dict[int, RidgeMap]) -> Dict[str, float]:
    rec = ev["conds"].get(("at", cond))
    if rec is None or ev["episode"] not in recs["refs"]:
        return {"cross": np.nan, "same": np.nan, "direct": np.nan}
    Rc = _refs_centred(recs, ev["episode"], li)
    t = ev["target"]
    cross, same, direct = [], [], []
    for c in unique_order_cells(rec["order"]):
        x = np.asarray(rec["h"][marker][c, li], np.float64)
        j = int(np.where(rec["order"][c] == t)[0][0])
        direct.append(selection_index(x, Rc, t))
        if j in maps:
            same.append(selection_index(maps[j].predict(x), Rc, t))
        for i, m in maps.items():
            if i != j:
                # A map fitted at position i reads whatever goal sits at i,
                # here a NON-target whose evidence the clause lowers. Left in
                # the comparison set, that alone makes present-minus-neutral
                # positive under a purely position-bound code (found with the
                # null simulation). So the goal at position i is excluded.
                cross.append(selection_index(m.predict(x), Rc, t, exclude=int(rec["order"][c][i])))
    mean = lambda v: float(np.mean(v)) if v else float("nan")
    return {"cross": mean(cross), "same": mean(same), "direct": mean(direct)}


def selection_effects(recs, train_families: Sequence[str], marker: str, li: int, min_dom: float = 0.5,
                      alpha: float = 1.0, min_rows: int = 4) -> Dict[str, Dict[str, float]]:
    """Per event: SI(a) - SI(b) for every contrast and every flavour of the
    index, plus the behavioural clause effect. Maps are fitted on
    `train_families` only."""
    X, Y, pos = _training_rows(recs, train_families, marker, li, min_dom)
    maps = {int(p): RidgeMap(X[pos == p], Y[pos == p], alpha)
            for p in np.unique(pos) if (pos == p).sum() >= min_rows}
    out = {}
    for eid, ev in recs["events"].items():
        s = {c: _event_si(recs, ev, c, marker, li, maps) for c in EVENT_CONDS}
        row = {}
        for flavour in ("cross", "same", "direct"):
            for name, (a, b) in CONTRASTS.items():
                row[f"{flavour}:{name}"] = s[a][flavour] - s[b][flavour]
            row[f"{flavour}:present"] = s["present"][flavour]
        t = ev["target"]
        beh = lambda c: float(ev["conds"][("at", c)]["dist"][:, t].mean()) if ("at", c) in ev["conds"] else float("nan")
        row["behaviour:present_minus_neutral_primary"] = beh("present") - beh("neutral_primary")
        row["behaviour:cue_minus_neutral_primary"] = beh("cue") - beh("neutral_primary")
        row["n_train_rows"] = int(len(pos))
        row["n_maps"] = len(maps)
        out[eid] = row
    return out


PRIMARY = "cross:present_minus_neutral_primary"
INERT = "cross:present_minus_inert_neutral"
CUE = "cross:cue_minus_neutral_primary"
LEXICAL = "cross:ctl_present_minus_ctl_neutral"
ALPHA = 0.05


def analyse_selection(recs, marker: str = "post_list", seed: int = 0, min_dom: float = 0.5,
                      alpha: float = 1.0, layer_index: Optional[int] = None) -> Dict[str, object]:
    """Full G1 analysis: fit on train families, choose the layer on
    development families, report the held-out test families once."""
    events = recs["events"]
    fam_of = {e: v["family"] for e, v in events.items()}
    train, dev, test = family_split(list(fam_of.values()), seed)
    some = next(iter(events.values()))["conds"]
    n_layers = next(iter(some.values()))["h"][marker].shape[1]
    per_layer = {li: selection_effects(recs, train, marker, li, min_dom, alpha) for li in range(n_layers)}

    def mean_over(li, fams, key):
        v = [per_layer[li][e][key] for e in events if fam_of[e] in fams]
        v = [x for x in v if not np.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    dev_curve = {li: mean_over(li, dev, PRIMARY) for li in range(n_layers)}
    if layer_index is None:
        finite = {k: v for k, v in dev_curve.items() if not np.isnan(v)}
        layer_index = max(finite, key=finite.get) if finite else n_layers // 2
    test_ev = [e for e in events if fam_of[e] in test]
    clusters = [fam_of[e] for e in test_ev]
    at = per_layer[layer_index]
    res = {"marker": marker, "families": {"train": train, "dev": dev, "test": test},
           "selected_layer_index": int(layer_index), "dev_curve_primary": dev_curve,
           "n_test_events": len(test_ev), "n_test_families": len(set(clusters)), "test": {}, "per_event": {}}
    keys = [k for k in at[next(iter(events))] if ":" in k]
    for k in keys:
        res["test"][k] = cluster_bootstrap([at[e][k] for e in test_ev], clusters, seed=seed)
    res["test"]["cross:trigger_minus_inert_edit"] = cluster_bootstrap(
        [at[e][PRIMARY] - at[e][INERT] for e in test_ev], clusters, seed=seed)
    res["test"]["cross:trigger_minus_lexical_control"] = cluster_bootstrap(
        [at[e][PRIMARY] - at[e][LEXICAL] for e in test_ev], clusters, seed=seed)
    res["per_event"] = {e: dict(at[e], family=fam_of[e], kind=events[e]["kind"],
                                split="train" if fam_of[e] in train else "dev" if fam_of[e] in dev else "test")
                        for e in events}
    sig = lambda k: res["test"][k]["p_signflip"] is not None and res["test"][k]["p_signflip"] <= ALPHA \
        and (res["test"][k]["mean"] or 0) > 0
    res["enough_families_to_test"] = res["n_test_families"] >= MIN_TEST_FAMILIES
    res["positive_control_detected"] = sig(CUE) if res["test"][CUE]["n_events"] else None
    lex = res["test"]["cross:trigger_minus_lexical_control"]
    res["exceeds_lexical_control"] = sig("cross:trigger_minus_lexical_control") if lex["n_events"] else None
    res["g1_supported"] = bool(sig(PRIMARY) and sig("cross:trigger_minus_inert_edit"))
    res["test_curve_primary"] = {li: mean_over(li, test, PRIMARY) for li in range(n_layers)}
    res["test_curve_cue"] = {li: mean_over(li, test, CUE) for li in range(n_layers)}
    return res


# ------------------------------------------------- candidate-token analyses
def candidate_samples(recs, families: Sequence[str], li: int, min_dom: float = 0.5):
    """One sample per (prefix, distinct order, list position): activation at
    that candidate's last token, y = this candidate is the dominant goal."""
    fams = set(families)
    X, y, pos, grp, is_true, dom_true, fam = [], [], [], [], [], [], []
    g = 0
    for r in recs["traj"]:
        if r["family"] not in fams:
            continue
        pbar = r["dist"].mean(0)
        dom = int(np.argmax(pbar))
        if pbar[dom] < min_dom:
            continue
        stored = r.get("cand_cells") or list(range(len(r["h_cand"])))
        for c in unique_order_cells(r["order"]):
            if c not in stored:
                continue
            row = stored.index(c)
            for k in range(4):
                gi = int(r["order"][c][k])
                X.append(r["h_cand"][row, k, li]); y.append(int(gi == dom)); pos.append(k); grp.append(g)
                is_true.append(gi == r["true"]); dom_true.append(dom == r["true"]); fam.append(r["family"])
            g += 1
    return (np.asarray(X, np.float32), np.asarray(y), np.asarray(pos), np.asarray(grp),
            np.asarray(is_true), np.asarray(dom_true), np.asarray(fam))


def value_probe_and_truth_transfer(recs, train: Sequence[str], heldout: Sequence[str], li: int,
                                   min_dom: float = 0.5) -> Dict[str, object]:
    """(1) Shared 'this candidate is dominant' probe, family-held-out.
    (2) G2 test (a): fit the probe ONLY on prefixes whose dominant goal is
    the true one, apply it to held-out prefixes whose dominant goal is WRONG.
    If the code is truth-blind the probe's top candidate is the (wrong)
    dominant goal; if it tracks truth it is the true goal; chance is 0.25."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    Xa, ya, _, _, _, dta, _ = candidate_samples(recs, train, li, min_dom)
    Xb, yb, _, gb, itb, dtb, _ = candidate_samples(recs, heldout, li, min_dom)
    out: Dict[str, object] = {"n_train": int(len(ya)), "n_heldout": int(len(yb))}
    if len(ya) < 16 or len(yb) < 8 or len(np.unique(ya)) < 2:
        return out
    fit = lambda X, y: (lambda sc: (sc, LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
                                    .fit(sc.transform(X), y)))(StandardScaler().fit(X))
    sc, clf = fit(Xa, ya)
    out["dominance_balanced_accuracy"] = float(balanced_accuracy_score(yb, clf.predict(sc.transform(Xb))))
    if dta.sum() >= 16 and (~dtb).sum() >= 8:
        sc, clf = fit(Xa[dta], ya[dta])
        score = clf.decision_function(sc.transform(Xb))
        top_dom, top_true, n = 0, 0, 0
        for g in np.unique(gb[~dtb]):
            idx = np.where(gb == g)[0]
            k = idx[np.argmax(score[idx])]
            top_dom += int(yb[k] == 1); top_true += int(itb[k]); n += 1
        out["wrong_commitments"] = {"n_prefix_cells": n, "top_is_dominant_wrong_goal": top_dom / n,
                                    "top_is_true_goal": top_true / n, "chance": 0.25}
    return out


def dominance_parallelism(recs, families: Sequence[str], li: int, min_dom: float = 0.5,
                          n_shuffle: int = 200, seed: int = 0) -> Dict[str, object]:
    """Is the dominant-vs-rejected axis the same at every list position?
    Coding vector at position k = mean(dominant at k) - mean(rejected at k);
    parallelism = mean cosine between positions. Null: permute the dominant
    label among the four candidates of each prompt."""
    from .geometry import parallelism_score
    X, y, pos, grp, _, _, _ = candidate_samples(recs, families, li, min_dom)
    if len(y) < 32:
        return {"n": int(len(y))}
    X = X.astype(np.float64)
    stat = lambda lab: parallelism_score(X, lab, np.array([f"{p}_{l}" for p, l in zip(pos, lab)]), pos)
    obs = stat(y)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_shuffle):
        lab = y.copy()
        for g in np.unique(grp):
            idx = np.where(grp == g)[0]
            lab[idx] = rng.permutation(lab[idx])
        null.append(stat(lab))
    null = np.asarray(null)
    return {"n": int(len(y)), "parallelism": float(obs), "null_mean": float(np.nanmean(null)),
            "p_one_sided": float((np.sum(null >= obs) + 1) / (n_shuffle + 1))}
