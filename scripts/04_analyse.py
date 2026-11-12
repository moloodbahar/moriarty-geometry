"""Stage 3 analysis on recorded activations.

G1  cross-position selection index, clause present vs neutral replacement
    (analysis/selection.py). Maps fitted on training families, layer chosen
    on development families, held-out families tested once with a
    family-level sign-flip test.
    Read with its two controls: the explicit-cue positive control (is the
    instrument sensitive on this model at all?) and the control-task lexical
    control (is the clause effect more than word overlap?).
G2a truth-blind transfer of the candidate-level dominance probe, and
    parallelism of the dominant-vs-rejected axis across list positions.

The output is labelled DEVELOPMENT ONLY unless the registry is frozen and
there are at least 6 held-out families and 15 held-out events.
"""
import argparse, hashlib, json, sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moriarty_geometry.analysis.selection import (analyse_selection, load_recordings, PRIMARY, CUE, MIN_TEST_FAMILIES,
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

MARKERS = [m for m in ("post_list", "after_candidates", "answer") if m in rec0["h"]]
for marker in MARKERS:
    r = analyse_selection(recs, marker, args.seed, args.min_dom, layer_index=args.layer_index)
    r["selected_layer"] = layers[r["selected_layer_index"]]
    res[f"G1:{marker}"] = r

g1 = res[f"G1:{MARKERS[0]}"]
fam = g1["families"]; li = g1["selected_layer_index"]
res["G2a:candidate_tokens"] = value_probe_and_truth_transfer(recs, fam["train"], fam["dev"] + fam["test"], li, args.min_dom)
res["G2a:parallelism"] = dominance_parallelism(recs, fam["dev"] + fam["test"], li, args.min_dom, seed=args.seed)
ok = frozen and g1["n_test_events"] >= 15 and g1["n_test_families"] >= MIN_TEST_FAMILIES
res["status"] = "confirmatory-eligible" if ok else "DEVELOPMENT ONLY"
pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1, default=float)

rnd = lambda x: None if x is None else round(x, 4)
print(f"status: {res['status']}  (events={res['n_events']}, held-out: {g1['n_test_events']} events in "
      f"{g1['n_test_families']} families, registry frozen={frozen})")
if g1["n_test_families"] < MIN_TEST_FAMILIES:
    print(f"  {g1['n_test_families']} held-out families: the sign-flip test cannot reach p <= 0.05 "
          f"(smallest attainable p = {2.0 ** -max(g1['n_test_families'], 1):.3f}). Read means and per-event values only.")
SHOW = [(PRIMARY, "trigger effect (PRIMARY)"), ("cross:present_minus_neutral_secondary", "same, second neutral wording"),
        ("cross:present_minus_inert_neutral", "inert-clause edit"), ("cross:trigger_minus_inert_edit", "trigger minus inert edit"),
        ("cross:ctl_present_minus_ctl_neutral", "lexical control (control task)"),
        ("cross:trigger_minus_lexical_control", "trigger minus lexical control"),
        (CUE, "positive control (explicit cue)"), ("same:cue_minus_neutral_primary", "positive control, same-position"),
        ("same:present_minus_neutral_primary", "same-position (not a test)"),
        ("behaviour:present_minus_neutral_primary", "behavioural clause effect"),
        ("behaviour:cue_minus_neutral_primary", "behavioural cue effect")]
for marker in MARKERS:
    r = res[f"G1:{marker}"]; t = r["test"]
    print(f"\nG1 at {marker}: layer {r['selected_layer']} (chosen on dev families {r['families']['dev']})")
    for k, label in SHOW:
        v = t[k]; ci = v["ci90"]
        print(f"  {label:34s} mean={rnd(v['mean'])}  p={rnd(v['p_signflip'])}  ci90={None if ci is None else (rnd(ci[0]), rnd(ci[1]))}  "
              f"n={v['n_events']} ev / {v['n_families']} fam")
    print(f"  positive control detected: {r['positive_control_detected']}   exceeds lexical control: {r['exceeds_lexical_control']}   "
          f"G1 supported: {r['g1_supported']}")
print("\nG2a candidate tokens:", json.dumps(res["G2a:candidate_tokens"], default=float))
print("G2a parallelism:", json.dumps(res["G2a:parallelism"], default=float))
print("\nwrote", args.out)
