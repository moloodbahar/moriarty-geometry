"""Stage 4: activation patching (G3 / H4).

Phase A — sanity. Must pass before any number below is read.
  identity      patching a prompt with its own activations changes nothing;
  last block    writing the clause-present run's last-block state at the
                answer position into the neutral run reproduces the
                clause-present answer exactly;
  restore curve the WHOLE shared suffix (everything after the story) written
                from present into neutral, block by block. Not expected to be
                1.0 at early blocks: later blocks still attend to the neutral
                clause tokens. It shows where the clause effect has moved
                into the suffix.

Phase C — cross-cell patch. THE PRIMARY CAUSAL TEST. No learned object.
  d = (present - neutral) at one read-out token of a SOURCE cell, added to
  the neutral run of a RECIPIENT cell of the same event.
  cross-order   recipient has a different list order (same mapping). The
                target goal sits at another list position there.
                  position-invariant content code -> the answer moves to the
                                                     TARGET goal;
                  position-bound code             -> it moves to the goal that
                                                     sits at the source's
                                                     target position.
                Reported: d_target, d_position_goal, and their difference, by
                block, at `post_list` (before the mapping) and at `answer`.
  cross-mapping recipient has a different answer mapping (same order), patch
                at `answer`: content code -> the answer moves to the target
                goal (now another letter); action code -> to the goal that
                owns the source's target letter. (Before the mapping the
                state is identical across mappings, so this is only
                informative at the answer position.)
  controls      a random vector of the same norm; the present-minus-inert-edit
                difference; and the same-cell patch as ceiling.
  The block is chosen on development families; held-out families are tested
  with a family-level sign-flip test.

Phase B — subspace patch at the answer position (the earlier H4 design),
  kept as a secondary analysis; see the docstring of analysis/patching.py.

An event enters phases B and C only if its clause raises the target goal by
--min-effect on THIS model in the decoupled format.
"""
import argparse, json, sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moriarty_geometry.prompts import build_decoupled, latin_orders, latin_mappings
from moriarty_geometry.probe import OpenWeightsProbe
from moriarty_geometry.neutral import load_registry, condition_texts
from moriarty_geometry.episodes import load_episodes
from moriarty_geometry.analysis.selection import family_of, family_split, cluster_bootstrap, MIN_TEST_FAMILIES

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--episodes", nargs="+", default=["data/episodes_confirmatory_reconstructed.json"])
ap.add_argument("--report", nargs="*", default=None, help="MORIARTY checks_report.json files; episodes they mark not usable are dropped")
ap.add_argument("--registry", default="data/neutral_registry.json")
ap.add_argument("--blocks", type=int, nargs="*", default=None, help="block indices; default: 8 evenly spaced")
ap.add_argument("--pairs", type=int, default=2, help="source->recipient pairs per event and kind (1-4)")
ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--n-random", type=int, default=1)
ap.add_argument("--min-effect", type=float, default=0.05, help="events whose clause raises the target goal by less than this are skipped")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sanity-only", action="store_true")
ap.add_argument("--skip-subspace", action="store_true", help="skip phase B")
ap.add_argument("--out", default="results/patching.json")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

import torch
conf = load_episodes(args.episodes, args.report)
probe = OpenWeightsProbe(args.model, device=args.device)
NB = probe.n_layers
blocks = args.blocks or sorted(set(int(round(x)) for x in np.linspace(1, NB - 1, min(8, NB - 1))))
ORD, MAP = latin_orders(), latin_mappings()
registry = load_registry(args.registry)
rng = np.random.default_rng(args.seed)
pg = lambda res, goal: float(res.goal_dist[goal])

class Cell:
    """One (order, mapping) cell of one event under the three conditions,
    with block outputs captured at the two read-out sites."""
    def __init__(self, before, texts, goals, oi, mi):
        self.oi, self.mi, self.order, self.goals = oi, mi, ORD[oi], goals
        self.spec = {k: build_decoupled(before + [t], goals, ORD[oi], MAP[mi]) for k, t in texts.items()}
        self.k = {"post_list": probe.offset_from_end(self.spec["N"], "post_list"), "answer": 1}
        self.same_offsets = len({probe.offset_from_end(s, "post_list") for s in self.spec.values()}) == 1
        self.n = min(probe.suffix_len(s) for s in self.spec.values())
        self.res, self.h = {}, {}
        for c, s in self.spec.items():
            self.res[c], cap = probe.run_blocks(s, self.k["post_list"], capture=blocks)
            self.h[c] = {site: {b: cap[b][-self.k[site]].numpy().astype(np.float64) for b in blocks} for site in self.k}
    def p(self, cond, goal): return pg(self.res[cond], goal)
    def diff(self, site, b, other="N"): return self.h["P"][site][b] - self.h[other][site][b]
    def letter_of(self, goal_text): return next(L for L, g in self.spec["N"].letter_to_goal.items() if g == goal_text)

def patched(cell, cond, site, b, delta):
    k = cell.k[site]; d = torch.from_numpy(np.asarray(delta, np.float32))
    def fn(h):
        h = h.clone(); h[0] += d; return h
    return probe.run_blocks(cell.spec[cond], k, edit=(b, fn))[0]

def rand_like(v):
    r = rng.standard_normal(v.shape); return r * np.linalg.norm(v) / np.linalg.norm(r)

# ------------------------------------------------------------- collect runs
events = []
for r in registry:
    ep = conf[r.episode_id]["episode"]; goals = ep["goals"]; step = ep["steps"][r.step - 1]
    ct = condition_texts(r, step)
    texts = {"P": ct["present"], "N": ct["neutral_primary"], "I": ct["inert_neutral"]}
    before = ep["steps"][: r.step - 1]
    need = {(o, 0) for o in range(4)} | {(0, m) for m in range(4)}
    cells = {om: Cell(before, texts, goals, *om) for om in sorted(need)}
    eff = float(np.mean([c.p("P", r.target_goal) - c.p("N", r.target_goal) for c in cells.values()]))
    events.append({"id": r.event_id, "family": family_of(r.episode_id), "target": r.target_goal,
                   "ti": goals.index(r.target_goal), "goals": goals, "cells": cells, "effect": eff})
    print(f"{r.event_id:34s} p_target present={np.mean([c.p('P', r.target_goal) for c in cells.values()]):.3f} "
          f"neutral={np.mean([c.p('N', r.target_goal) for c in cells.values()]):.3f} "
          f"inert={np.mean([c.p('I', r.target_goal) for c in cells.values()]):.3f}")

# ------------------------------------------------------------------ phase A
sanity = {"identity_max_abs_diff": 0.0, "last_block_max_abs_diff": 0.0, "restore_curve": {}}
for ev in events:
    c = ev["cells"][(0, 0)]; P, N, n, g = c.spec["P"], c.spec["N"], c.n, ev["target"]
    same, _ = probe.run_blocks(P, n, edit=(blocks[len(blocks) // 2], lambda h: h))
    sanity["identity_max_abs_diff"] = max(sanity["identity_max_abs_diff"], abs(pg(same, g) - c.p("P", g)))
    _, hP = probe.run_blocks(P, n, capture=[NB - 1])
    def last(h, d=hP[NB - 1]):
        h = h.clone(); h[-1] = d[-1]; return h
    got, _ = probe.run_blocks(N, n, edit=(NB - 1, last))
    sanity["last_block_max_abs_diff"] = max(sanity["last_block_max_abs_diff"], abs(pg(got, g) - c.p("P", g)))
    gap = c.p("P", g) - c.p("N", g)
    if abs(gap) >= args.min_effect:
        curve = {}
        for b in blocks:
            _, hPb = probe.run_blocks(P, n, capture=[b])
            got, _ = probe.run_blocks(N, n, edit=(b, lambda h, d=hPb[b]: d.clone()))
            curve[b] = (pg(got, g) - c.p("N", g)) / gap
        sanity["restore_curve"][ev["id"]] = curve
tol = 5e-3
sanity["pass"] = bool(sanity["identity_max_abs_diff"] < tol and sanity["last_block_max_abs_diff"] < tol)
print(f"\nsanity: identity {sanity['identity_max_abs_diff']:.2e}, last-block {sanity['last_block_max_abs_diff']:.2e} "
      f"-> {'PASS' if sanity['pass'] else 'FAIL'} (tolerance {tol}); prompt rendering: {getattr(probe, 'template_mode', '?')}")
for eid, curve in sanity["restore_curve"].items():
    print(f"  suffix restore {eid}: " + "  ".join(f"b{b}={v:+.2f}" for b, v in curve.items()))
result = {"model": args.model, "blocks": blocks, "sanity": sanity, "prompt_rendering": getattr(probe, "template_mode", None)}

def finish(msg=None):
    if msg: print(msg); result["note"] = msg
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1, default=float); print("wrote", args.out); sys.exit(0)

if args.sanity_only: finish()
if not sanity["pass"]: finish("sanity failed — patch numbers not computed")
usable = [e for e in events if e["effect"] >= args.min_effect]      # the clause must raise its target goal on THIS model
print(f"\n{len(usable)} of {len(events)} events have a behavioural clause effect >= {args.min_effect} in the decoupled format")
if not usable: finish("no event shows the clause effect on this model")

# ------------------------------------------------------------------ phase C
def cross_cell(ev, kind, site, b):
    """Mean over source->recipient pairs of the change in probability of
    (target goal, competitor goal) in the recipient, plus controls."""
    out = {k: [] for k in ("target", "competitor", "random_target", "inert_target", "same_cell_target")}
    ti, goals, cells = ev["ti"], ev["goals"], ev["cells"]
    for j in range(args.pairs):
        if kind == "order":
            S, R = cells[(j, 0)], cells[((j + 1) % 4, 0)]
            comp = goals[R.order[S.order.index(ti)]]            # goal at the source's target POSITION
        else:
            S, R = cells[(0, j)], cells[(0, (j + 1) % 4)]
            comp = R.spec["N"].letter_to_goal[S.letter_of(ev["target"])]   # goal owning the source's target LETTER
        if S.k[site] != R.k[site] or not (S.same_offsets and R.same_offsets):
            continue                                            # token alignment differs: skip, never guess
        d = S.diff(site, b); base_t, base_c = R.p("N", ev["target"]), R.p("N", comp)
        got = patched(R, "N", site, b, d)
        out["target"].append(pg(got, ev["target"]) - base_t); out["competitor"].append(pg(got, comp) - base_c)
        out["random_target"].append(float(np.mean([pg(patched(R, "N", site, b, rand_like(d)), ev["target"]) - base_t
                                                   for _ in range(args.n_random)])))
        out["inert_target"].append(pg(patched(R, "N", site, b, S.diff(site, b, "I")), ev["target"]) - base_t)
        out["same_cell_target"].append(pg(patched(S, "N", site, b, d), ev["target"]) - S.p("N", ev["target"]))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()} | {"n_pairs": len(out["target"])}

TESTS = [("order", "post_list"), ("order", "answer"), ("mapping", "answer")]
phaseC = {f"{k}@{s}": {e["id"]: {b: cross_cell(e, k, s, b) for b in blocks} for e in usable} for k, s in TESTS}
fams = sorted({e["family"] for e in usable})
if len(fams) >= 3:
    tr, dv, te = family_split(fams, args.seed); dev_f, test_f = set(tr) | set(dv), set(te)
else:
    dev_f, test_f = set(fams), set()
result["cross_cell"] = {"families": {"dev": sorted(dev_f), "test": sorted(test_f)}, "per_event": phaseC, "summary": {}}
fam_of_ev = {e["id"]: e["family"] for e in usable}
print("\nPhase C — cross-cell patch (change in probability in the recipient cell; all usable events, by block)")
for key, table in phaseC.items():
    kind, site = key.split("@")
    comp_name = "goal at source position" if kind == "order" else "goal with source letter"
    print(f"  {key}:  block: target / {comp_name} / random / inert / same-cell ceiling")
    for b in blocks:
        m = lambda f: np.nanmean([table[e][b][f] for e in table])
        print(f"    b{b:<3d} {m('target'):+.3f} / {m('competitor'):+.3f} / {m('random_target'):+.3f} / {m('inert_target'):+.3f} / {m('same_cell_target'):+.3f}")
    idx = lambda e, b: table[e][b]["target"] - table[e][b]["competitor"]
    dev_ev = [e for e in table if fam_of_ev[e] in dev_f]; test_ev = [e for e in table if fam_of_ev[e] in test_f]
    b_star = max(blocks, key=lambda b: np.nanmean([idx(e, b) for e in dev_ev])) if dev_ev else blocks[len(blocks) // 2]
    summ = {"selected_block": b_star, "n_test_families": len(test_f)}
    if test_ev:
        cl = [fam_of_ev[e] for e in test_ev]
        summ["target_minus_competitor"] = cluster_bootstrap([idx(e, b_star) for e in test_ev], cl, seed=args.seed)
        summ["target_minus_random"] = cluster_bootstrap([table[e][b_star]["target"] - table[e][b_star]["random_target"] for e in test_ev], cl, seed=args.seed)
        summ["target_minus_inert"] = cluster_bootstrap([table[e][b_star]["target"] - table[e][b_star]["inert_target"] for e in test_ev], cl, seed=args.seed)
        for k in ("target_minus_competitor", "target_minus_random", "target_minus_inert"):
            v = summ[k]; print(f"    held-out, block {b_star}: {k:24s} mean={v['mean']:+.3f} p={v['p_signflip']} n={v['n_events']} ev / {v['n_families']} fam")
    result["cross_cell"]["summary"][key] = summ
result["status"] = "confirmatory-eligible" if len(test_f) >= MIN_TEST_FAMILIES else \
    f"DEVELOPMENT ONLY ({len(test_f)} held-out families; {MIN_TEST_FAMILIES} needed for the test to reach p <= 0.05)"
print("status:", result["status"])

# ------------------------------------------------------------------ phase B
if args.skip_subspace or len(fams) < 3: finish()
train, dev, test = family_split(fams, args.seed)
cellsB = lambda e: [e["cells"][(o, 0)] for o in range(4)]
def diffs(e, b, other="N"): return np.stack([c.diff("answer", b, other) for c in cellsB(e)])
def basis(rows, rank): return np.linalg.svd(rows, full_matrices=False)[2][:rank].T
def rescale(v, like): return v * np.linalg.norm(like) / (np.linalg.norm(v) + 1e-12)

def evaluate(e, b, U, UI, donors, full_controls):
    out = {k: [] for k in ("restore", "remove", "full", "random", "other_donor", "inert_subspace", "norm_ratio")}
    g = e["target"]
    for c in cellsB(e):
        d = c.diff("answer", b); inj = U @ (U.T @ d)
        out["restore"].append(pg(patched(c, "N", "answer", b, inj), g) - c.p("N", g))
        hN = c.h["N"]["answer"][b]; out["norm_ratio"].append(float(np.linalg.norm(hN + inj) / np.linalg.norm(hN)))
        if not full_controls: continue
        out["remove"].append(pg(patched(c, "P", "answer", b, -inj), g) - c.p("P", g))
        out["full"].append(pg(patched(c, "N", "answer", b, d), g) - c.p("N", g))
        rnd = []
        for _ in range(args.n_random):
            R = np.linalg.qr(rng.standard_normal((len(d), U.shape[1])))[0]
            rnd.append(pg(patched(c, "N", "answer", b, rescale(R @ (R.T @ d), inj)), g) - c.p("N", g))
        out["random"].append(float(np.mean(rnd)))
        if donors:
            od = donors[rng.integers(len(donors))]
            out["other_donor"].append(pg(patched(c, "N", "answer", b, rescale(U @ (U.T @ od), inj)), g) - c.p("N", g))
        out["inert_subspace"].append(pg(patched(c, "N", "answer", b, UI @ (UI.T @ d)), g) - c.p("N", g))
    return {k: float(np.mean(v)) if v else None for k, v in out.items()}

trE = [e for e in usable if e["family"] in train]; dvE = [e for e in usable if e["family"] in dev]
teE = [e for e in usable if e["family"] in test]
grid = {}
for b in blocks:
    rows = np.concatenate([diffs(e, b) for e in trE])
    for r in args.ranks:
        if r > min(rows.shape): continue
        grid[(b, r)] = float(np.mean([evaluate(e, b, basis(rows, r), None, None, False)["restore"] for e in dvE]))
(b, r) = max(grid, key=grid.get)
print(f"\nPhase B — subspace patch at the answer position: block {b}, rank {r} chosen on dev families {dev} (dev restore {grid[(b, r)]:+.3f})")
U = basis(np.concatenate([diffs(e, b) for e in trE]), r)
rowsI = np.concatenate([diffs(e, b, "I") for e in trE]); UI = basis(rowsI, min(r, min(rowsI.shape)))
per_event = {}
for e in teE:
    donors = [diffs(x, b).mean(0) for x in usable if x["family"] != e["family"]]
    per_event[e["id"]] = dict(evaluate(e, b, U, UI, donors, True), family=e["family"], behavioural_effect=e["effect"])
fam = [v["family"] for v in per_event.values()]
col = lambda k: [v[k] for v in per_event.values()]
contrast = lambda a, c: cluster_bootstrap([x - y for x, y in zip(col(a), col(c))], fam, seed=args.seed)
result["subspace"] = {"families": {"train": train, "dev": dev, "test": test}, "selected": {"block": b, "rank": r},
                      "dev_grid": {f"{k[0]}:{k[1]}": v for k, v in grid.items()}, "per_event": per_event,
                      "test": {"restore_minus_random": contrast("restore", "random"),
                               "restore_minus_other_donor": contrast("restore", "other_donor"),
                               "restore_minus_inert_subspace": contrast("restore", "inert_subspace"),
                               "remove": cluster_bootstrap(col("remove"), fam, seed=args.seed),
                               "full": cluster_bootstrap(col("full"), fam, seed=args.seed),
                               "norm_ratio": cluster_bootstrap(col("norm_ratio"), fam, seed=args.seed)}}
for k, v in result["subspace"]["test"].items():
    print(f"  {k:30s} mean={v['mean']}  p={v['p_signflip']}  n={v['n_events']} ev / {v['n_families']} fam")
finish()
