"""Stage 3 analyses on recorded activations: cross-position content decoding
(Li analogue), CCGP / parallelism for the dominant-vs-rejected and
position dichotomies, and paired-difference patching with controls.
Splits are by goal family; the layer/rank selection is done on the
development families and frozen before the held-out evaluation."""
import argparse, glob, json, sys, pathlib, re
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from moriarty_geometry.analysis.geometry import ccgp, parallelism_score, cross_position_generalisation, shuffle_null
from moriarty_geometry.analysis.patching import learn_subspace, patch_restore, patch_remove, matched_random_patch

ap = argparse.ArgumentParser()
ap.add_argument("--activations", default="results/activations")
ap.add_argument("--marker", default="after_candidates")
ap.add_argument("--layer", type=int, default=None, help="if omitted, chosen on dev families")
ap.add_argument("--out", default="results/analysis.json")
args = ap.parse_args()

files = sorted(glob.glob(f"{args.activations}/*__at__present__*.npz"))
rows = []
for f in files:
    ev = pathlib.Path(f).name.split("__")[0]; fam = ev.split("_")[0]
    z = np.load(f, allow_pickle=True)
    h = z[f"h_{args.marker}"]            # (cells, layers, d)
    dist = z["dist"]; order = z["order"]
    for c in range(h.shape[0]):
        dom = int(np.argmax(dist[c]))
        for pos, gi in enumerate(order[c]):
            rows.append(dict(event=ev, family=fam, cell=c, layer_acts=h[c], goal=int(gi), pos=pos, dominant=int(gi == dom)))
if not rows:
    sys.exit("no activations found")
fams = sorted({r["family"] for r in rows}); dev = set(fams[: max(1, len(fams) // 3)]); test = set(fams) - dev
L = rows[0]["layer_acts"].shape[0]
def layer_stat(layer, subset):
    X = np.stack([r["layer_acts"][layer] for r in rows if r["family"] in subset])
    y = np.array([r["dominant"] for r in rows if r["family"] in subset])
    cond = np.array([f"{r['goal']}_{r['pos']}" for r in rows if r["family"] in subset])
    pk = np.array([str(r["goal"]) for r in rows if r["family"] in subset])
    return ccgp(X, y, cond), parallelism_score(X, y, cond, pk)
if args.layer is None:
    curve = {l: layer_stat(l, dev)[0] for l in range(L)}
    layer = max(curve, key=curve.get)
else:
    curve, layer = {}, args.layer
res = {"selected_layer": layer, "dev_curve": curve}
res["test_ccgp_dominant"], res["test_parallelism_dominant"] = layer_stat(layer, test)
json.dump(res, open(args.out, "w"), indent=1, default=float); print(res)
