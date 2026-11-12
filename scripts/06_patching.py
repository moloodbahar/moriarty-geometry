"""Stage 4: paired activation patching (G3 / H4).

Phase A — sanity, must pass before any number below is read:
  identity      patching a prompt with its own activations changes nothing;
  last block    writing the clause-present run's last-block state at the
                answer position into the neutral run reproduces the
                clause-present answer exactly;
  restore curve writing the WHOLE shared suffix (everything after the story)
                from present into neutral, block by block. This is not
                expected to be 100% at early blocks: later blocks still
                attend back to the neutral clause tokens. The curve shows at
                which depth the clause effect has moved into the suffix.

Phase B — subspace patch at the last `--tail` positions:
  U is learned (uncentred SVD) from present-minus-neutral differences of
  TRAINING families; (block, rank) is chosen on DEVELOPMENT families; the
  held-out families are evaluated once.
    restore        neutral + U U^T (present - neutral)
    remove         present - U U^T (present - neutral)
    random         same rank, random subspace, injection rescaled to the same norm
    other_donor    another family's event difference through U, same norm
                   (if this works as well, the subspace is not content-specific)
    inert_subspace a subspace learned from present-minus-INERT-edit differences
                   (if this works as well, the subspace is a generic edit direction)
    full           the whole difference at those positions (ceiling)

A patch at the answer position in late blocks is close to writing the answer
itself; read the selected block against the restore curve.
"""
import argparse, json, sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moriarty_geometry.prompts import build_decoupled, decoupled_design
from moriarty_geometry.probe import OpenWeightsProbe
from moriarty_geometry.clauses import split_spans, edit_step
from moriarty_geometry.neutral import load_registry
from moriarty_geometry.analysis.selection import family_of, family_split, cluster_bootstrap

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--episodes", default="data/episodes_confirmatory_reconstructed.json")
ap.add_argument("--registry", default="data/neutral_registry.json")
ap.add_argument("--cells", type=int, default=4, help="design cells per event (1-4, one per list order)")
ap.add_argument("--tail", type=int, default=1, help="number of final positions patched in phase B (1 = answer position)")
ap.add_argument("--blocks", type=int, nargs="*", default=None, help="block indices; default: 8 evenly spaced")
ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--n-random", type=int, default=5)
ap.add_argument("--min-effect", type=float, default=0.05, help="events whose clause raises the target goal by less than this are skipped")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sanity-only", action="store_true")
ap.add_argument("--out", default="results/patching.json")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

import torch
conf = json.load(open(args.episodes))
probe = OpenWeightsProbe(args.model, device=args.device)
NB = probe.n_layers
blocks = args.blocks or sorted(set(int(round(x)) for x in np.linspace(1, NB - 1, min(8, NB - 1))))
design = [decoupled_design()[i] for i in (0, 5, 10, 15)[: args.cells]]
registry = load_registry(args.registry)

def p_target(res, goal): return float(res.goal_dist[goal])

# ------------------------------------------------------------- collect runs
events = []
for r in registry:
    ep = conf[r.episode_id]["episode"]; goals = ep["goals"]; step = ep["steps"][r.step - 1]
    spans = split_spans(step); before = ep["steps"][: r.step - 1]
    texts = {"P": step, "N": edit_step(step, spans, r.clause_index, r.neutral_primary),
             "I": edit_step(step, spans, r.inert_index, r.inert_neutral)}
    cells = []
    for o, m in design:
        spec = {k: build_decoupled(before + [t], goals, o, m) for k, t in texts.items()}
        n = min(probe.suffix_len(s) for s in spec.values())
        run = {k: probe.run_blocks(s, n, capture=blocks) for k, s in spec.items()}
        cells.append({"spec": spec, "n": n, "p": {k: p_target(v[0], r.target_goal) for k, v in run.items()},
                      "tail": {k: {b: v[1][b][-args.tail:].numpy().astype(np.float64) for b in blocks} for k, v in run.items()}})
    eff = float(np.mean([c["p"]["P"] - c["p"]["N"] for c in cells]))
    events.append({"id": r.event_id, "family": family_of(r.episode_id), "target": r.target_goal, "cells": cells, "effect": eff})
    print(f"{r.event_id:32s} p_target present={np.mean([c['p']['P'] for c in cells]):.3f} "
          f"neutral={np.mean([c['p']['N'] for c in cells]):.3f} inert={np.mean([c['p']['I'] for c in cells]):.3f}")

# ------------------------------------------------------------------ phase A
sanity = {"identity_max_abs_diff": 0.0, "last_block_max_abs_diff": 0.0, "restore_curve": {}}
for ev in events:
    c = ev["cells"][0]; P, N, n = c["spec"]["P"], c["spec"]["N"], c["n"]
    same, _ = probe.run_blocks(P, n, edit=(blocks[len(blocks) // 2], lambda h: h))
    sanity["identity_max_abs_diff"] = max(sanity["identity_max_abs_diff"], abs(p_target(same, ev["target"]) - c["p"]["P"]))
    _, hP = probe.run_blocks(P, n, capture=[NB - 1])
    def last(h, d=hP[NB - 1]):
        h = h.clone(); h[-1] = d[-1]; return h
    got, _ = probe.run_blocks(N, n, edit=(NB - 1, last))
    sanity["last_block_max_abs_diff"] = max(sanity["last_block_max_abs_diff"], abs(p_target(got, ev["target"]) - c["p"]["P"]))
    gap = c["p"]["P"] - c["p"]["N"]
    if abs(gap) >= args.min_effect:
        curve = {}
        for b in blocks:
            _, hPb = probe.run_blocks(P, n, capture=[b])
            got, _ = probe.run_blocks(N, n, edit=(b, lambda h, d=hPb[b]: d.clone()))
            curve[b] = (p_target(got, ev["target"]) - c["p"]["N"]) / gap
        sanity["restore_curve"][ev["id"]] = curve
tol = 5e-3
sanity["pass"] = bool(sanity["identity_max_abs_diff"] < tol and sanity["last_block_max_abs_diff"] < tol)
print(f"\nsanity: identity {sanity['identity_max_abs_diff']:.2e}, last-block {sanity['last_block_max_abs_diff']:.2e} "
      f"-> {'PASS' if sanity['pass'] else 'FAIL'} (tolerance {tol})")
for eid, curve in sanity["restore_curve"].items():
    print(f"  suffix restore {eid}: " + "  ".join(f"b{b}={v:+.2f}" for b, v in curve.items()))
result = {"model": args.model, "blocks": blocks, "tail": args.tail, "sanity": sanity}

def finish(msg=None):
    if msg: print(msg); result["note"] = msg
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1, default=float); print("wrote", args.out); sys.exit(0)

if args.sanity_only: finish()
if not sanity["pass"]: finish("sanity failed — patch numbers not computed")
usable = [e for e in events if e["effect"] >= args.min_effect]      # the clause must raise its target goal on THIS model
if len({e["family"] for e in usable}) < 3:
    finish(f"only {len({e['family'] for e in usable})} families have a behavioural clause effect >= {args.min_effect}; "
           "need >= 3 for a train/dev/test split")

# ------------------------------------------------------------------ phase B
train, dev, test = family_split([e["family"] for e in usable], args.seed)
rng = np.random.default_rng(args.seed)
def diffs(e, b, other="N"): return np.concatenate([c["tail"]["P"][b] - c["tail"][other][b] for c in e["cells"]])
def basis(rows, rank): return np.linalg.svd(rows, full_matrices=False)[2][:rank].T          # (d, rank)
def rand_basis(d, rank): return np.linalg.qr(rng.standard_normal((d, rank)))[0]
def rescale(v, like): return v * (np.linalg.norm(like, axis=1, keepdims=True) / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12))

def run_patch(spec, n, b, delta):
    d = torch.from_numpy(np.asarray(delta, np.float32))
    def fn(h):
        h = h.clone(); h[-args.tail:] += d; return h
    return probe.run_blocks(spec, n, edit=(b, fn))[0]

def evaluate(e, b, U, UI, donors, full_controls):
    out = {k: [] for k in ("restore", "remove", "full", "random", "other_donor", "inert_subspace", "norm_ratio")}
    for c in e["cells"]:
        d = c["tail"]["P"][b] - c["tail"]["N"][b]; inj = d @ U @ U.T; n = c["n"]; g = e["target"]
        out["restore"].append(p_target(run_patch(c["spec"]["N"], n, b, inj), g) - c["p"]["N"])
        out["norm_ratio"].append(float(np.linalg.norm(c["tail"]["N"][b] + inj) / np.linalg.norm(c["tail"]["N"][b])))
        if not full_controls: continue
        out["remove"].append(p_target(run_patch(c["spec"]["P"], n, b, -inj), g) - c["p"]["P"])
        out["full"].append(p_target(run_patch(c["spec"]["N"], n, b, d), g) - c["p"]["N"])
        rnd = []
        for _ in range(args.n_random):
            R = rand_basis(d.shape[1], U.shape[1])
            rnd.append(p_target(run_patch(c["spec"]["N"], n, b, rescale(d @ R @ R.T, inj)), g) - c["p"]["N"])
        out["random"].append(float(np.mean(rnd)))
        if donors:
            od = donors[rng.integers(len(donors))]
            out["other_donor"].append(p_target(run_patch(c["spec"]["N"], n, b, rescale(np.tile(od @ U @ U.T, (len(d), 1)), inj)), g) - c["p"]["N"])
        out["inert_subspace"].append(p_target(run_patch(c["spec"]["N"], n, b, d @ UI @ UI.T), g) - c["p"]["N"])
    return {k: float(np.mean(v)) if v else None for k, v in out.items()}

tr = [e for e in usable if e["family"] in train]; dv = [e for e in usable if e["family"] in dev]
te = [e for e in usable if e["family"] in test]
grid = {}
for b in blocks:
    rows = np.concatenate([diffs(e, b) for e in tr])
    for r in args.ranks:
        if r > min(rows.shape): continue
        U = basis(rows, r)
        grid[(b, r)] = float(np.mean([evaluate(e, b, U, None, None, False)["restore"] for e in dv]))
(b, r) = max(grid, key=grid.get)
print(f"\nselected on dev families {dev}: block {b}, rank {r} (dev restore {grid[(b, r)]:+.3f})")
U = basis(np.concatenate([diffs(e, b) for e in tr]), r)
UI = basis(np.concatenate([diffs(e, b, "I") for e in tr]), min(r, len(tr) * args.cells * args.tail))
per_event = {}
for e in te:
    donors = [diffs(x, b).mean(0) for x in usable if x["family"] != e["family"]]
    per_event[e["id"]] = dict(evaluate(e, b, U, UI, donors, True), family=e["family"], behavioural_effect=e["effect"])
fam = [v["family"] for v in per_event.values()]
col = lambda k: [v[k] for v in per_event.values()]
contrast = lambda a, c: cluster_bootstrap([x - y for x, y in zip(col(a), col(c))], fam, seed=args.seed)
result.update({"families": {"train": train, "dev": dev, "test": test}, "selected": {"block": b, "rank": r},
               "dev_grid": {f"{k[0]}:{k[1]}": v for k, v in grid.items()}, "per_event": per_event,
               "test": {"restore_minus_random": contrast("restore", "random"),
                        "restore_minus_other_donor": contrast("restore", "other_donor"),
                        "restore_minus_inert_subspace": contrast("restore", "inert_subspace"),
                        "remove": cluster_bootstrap(col("remove"), fam, seed=args.seed),
                        "full": cluster_bootstrap(col("full"), fam, seed=args.seed),
                        "norm_ratio": cluster_bootstrap(col("norm_ratio"), fam, seed=args.seed)}})
for k, v in result["test"].items():
    print(f"  {k:30s} mean={v['mean']}  ci90={v['ci90']}  n={v['n_events']} events / {v['n_families']} families")
rr = result["test"]["restore_minus_random"]
result["g3_criterion_met"] = bool(rr["ci90"] and rr["mean"] >= 0.15 and rr["ci90"][0] > 0 and len(te) >= 15)
result["status"] = "confirmatory-eligible" if len(te) >= 15 else "DEVELOPMENT ONLY (fewer than 15 held-out events)"
print("status:", result["status"])
finish()
