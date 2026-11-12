"""Stage 3 analysis on recorded activations.

G1  selection index, clause present vs neutral replacement, cross-position
    (analysis/selection.py). Maps fitted on training families, layer chosen
    on development families, held-out families reported once.
G2a truth-blind transfer of the candidate-level dominance probe, and
    parallelism of the dominant-vs-rejected axis across list positions.

Everything is aggregated per event and bootstrapped over families. With
fewer than 15 held-out events, or an unfrozen registry, the output is
development material and is labelled so.
"""
import argparse, hashlib, json, sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moriarty_geometry.analysis.selection import (analyse_selection, load_recordings, PRIMARY,
                                                    value_probe_and_truth_transfer, dominance_parallelism)

ap = argparse.ArgumentParser()
ap.add_argument("--activations", default="results/activations")
ap.add_argument("--format", default="decoupled")
ap.add_argument("--registry", default=str(ROOT / "data" / "neutral_registry.json"))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--min-dom", type=float, default=0.5, help="a prefix counts as having a dominant goal above this")
ap.add_argument("--layer-index", type=int, default=None, help="skip selection; index into the recorded layers")
ap.add_argument("--out", default="results/analysis.json")
args = ap.parse_args()

recs = load_recordings(args.activations, args.format)
if not recs["events"]:
    sys.exit("no event recordings found — run scripts/03_record_activations.py first")
if not recs["refs"]:
    sys.exit("no reference vectors found (ref__*.npz) — re-run scripts/03_record_activations.py")
rec0 = next(iter(next(iter(recs["events"].values()))["conds"].values()))
layers = [int(x) for x in rec0["layers"]]

res = {"preregistration_sha256": hashlib.sha256((ROOT / "PREREGISTRATION.md").read_bytes()).hexdigest(),
       "n_events": len(recs["events"]), "n_trajectory_prefixes": len(recs["traj"]), "recorded_layers": layers}
try:
    frozen = all(r.get("status") == "FROZEN" for r in json.load(open(args.registry))["records"])
except Exception:
    frozen = False

for marker in ("after_candidates", "answer"):
    r = analyse_selection(recs, marker, args.seed, args.min_dom, layer_index=args.layer_index)
    r["selected_layer"] = layers[r["selected_layer_index"]]
    res[f"G1:{marker}"] = r

g1 = res["G1:after_candidates"]
fam = g1["families"]; li = g1["selected_layer_index"]
res["G2a:candidate_tokens"] = value_probe_and_truth_transfer(recs, fam["train"], fam["dev"] + fam["test"], li, args.min_dom)
res["G2a:parallelism"] = dominance_parallelism(recs, fam["dev"] + fam["test"], li, args.min_dom, seed=args.seed)
res["status"] = "confirmatory-eligible" if (frozen and g1["n_test_events"] >= 15) else "DEVELOPMENT ONLY"
pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1, default=float)

print(f"status: {res['status']}  (events={res['n_events']}, held-out events={g1['n_test_events']}, registry frozen={frozen})")
for marker in ("after_candidates", "answer"):
    r = res[f"G1:{marker}"]; t = r["test"]
    print(f"\nG1 at {marker}: layer {r['selected_layer']} (chosen on dev families {r['families']['dev']})")
    for k in (PRIMARY, "cross:present_minus_neutral_secondary", "cross:present_minus_inert_neutral",
              "cross:trigger_minus_inert_edit", "same:present_minus_neutral_primary",
              "direct:present_minus_neutral_primary"):
        v = t[k]; ci = v["ci90"]
        print(f"  {k:42s} mean={v['mean'] if v['mean'] is None else round(v['mean'], 4)}  "
              f"ci90={None if ci is None else (round(ci[0], 4), round(ci[1], 4))}  n={v['n_events']} events / {v['n_families']} families")
    print(f"  supported (both one-sided lower bounds > 0): {r['g1_supported']}")
print("\nG2a candidate tokens:", json.dumps(res["G2a:candidate_tokens"], default=float))
print("G2a parallelism:", json.dumps(res["G2a:parallelism"], default=float))
print("\nwrote", args.out)
