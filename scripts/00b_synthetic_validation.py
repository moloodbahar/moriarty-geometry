"""Step 0. Validate the selection analysis on simulated recordings with a
known answer BEFORE trusting it on a model.

Three checks, each must pass:
  1. a planted position-invariant selected code is detected (power);
  2. a null in which the clause changes behaviour and the position-bound
     code, but no selected code exists, is NOT detected (false positives);
  3. the on-disk format round-trips through scripts/04_analyse.py's loader.

Set --families to the number of event-bearing families you plan to analyse:
the power figure is only meaningful at your planned n.
"""
import argparse, sys, pathlib, tempfile
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from moriarty_geometry.analysis.simulate import simulate, write
from moriarty_geometry.analysis.selection import analyse_selection, load_recordings, PRIMARY

ap = argparse.ArgumentParser()
ap.add_argument("--families", type=int, default=12)
ap.add_argument("--sims", type=int, default=40)
ap.add_argument("--strength", type=float, default=1.0)
ap.add_argument("--dim", type=int, default=96)
a = ap.parse_args()

def rate(strength):
    hits, means = [], []
    for s in range(a.sims):
        r = analyse_selection(simulate(a.families, a.dim, strength, seed=s), seed=s)
        hits.append(r["g1_supported"]); means.append(r["test"][PRIMARY]["mean"])
    return float(np.mean(hits)), float(np.mean(means))

power, m1 = rate(a.strength)
fpr, m0 = rate(0.0)
with tempfile.TemporaryDirectory() as d:
    recs = simulate(max(a.families, 3), a.dim, a.strength, seed=0)
    write(recs, d)
    back = load_recordings(d)
    r1, r2 = analyse_selection(recs, seed=0), analyse_selection(back, seed=0)
    roundtrip = abs(r1["test"][PRIMARY]["mean"] - r2["test"][PRIMARY]["mean"]) < 0.02 \
        and len(back["events"]) == len(recs["events"])

checks = [("planted effect detected (power >= 0.80)", power >= 0.80, f"power={power:.2f}, mean effect={m1:+.3f}"),
          ("null not detected (rate <= 0.05)", fpr <= 0.05, f"false-positive rate={fpr:.2f}, mean effect={m0:+.3f}"),
          ("on-disk format round-trips", roundtrip, "")]
print(f"families={a.families}  sims={a.sims}  strength={a.strength}")
for name, ok, info in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {info}")
sys.exit(0 if all(ok for _, ok, _ in checks) else 1)
