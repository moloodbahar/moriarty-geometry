"""Step 0. Validate the selection analysis on simulated recordings with a
known answer BEFORE trusting it on a model.

Checks, each must pass:
  1. power       a planted position-invariant selected code is detected;
  2. null        a clause that changes behaviour and the position-bound code,
                 with no selected code, is NOT detected;
  3. calibration under an exact null (two independent draws of one condition)
                 the family sign-flip test rejects at about its nominal rate;
  4. round trip  the on-disk format loads back to the same result.

Set --families to the number of event-bearing families you plan to analyse.
With the 50/20/30 split, fewer than 20 families leaves fewer than 6 held-out
families, and the test then cannot reach p <= 0.05 at all: check 1 will fail,
which is the correct answer at that n.
"""
import argparse, sys, pathlib, tempfile
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from moriarty_geometry.analysis.simulate import simulate, write
from moriarty_geometry.analysis.selection import analyse_selection, load_recordings, PRIMARY, INERT, ALPHA

ap = argparse.ArgumentParser()
ap.add_argument("--families", type=int, default=24)
ap.add_argument("--sims", type=int, default=40)
ap.add_argument("--strength", type=float, default=1.0)
ap.add_argument("--dim", type=int, default=64)
a = ap.parse_args()

def run(strength):
    g1, exact, means, nfam = [], [], [], 0
    for s in range(a.sims):
        r = analyse_selection(simulate(a.families, a.dim, strength, seed=s, inert_peak=0.82), seed=s)
        g1.append(r["g1_supported"]); means.append(r["test"][PRIMARY]["mean"]); nfam = r["n_test_families"]
        p = r["test"][INERT]["p_signflip"]; exact.append(p is not None and p <= ALPHA)
    return float(np.mean(g1)), float(np.mean(means)), exact, nfam

power, m1, ex1, nfam = run(a.strength)
fpr, m0, ex0, _ = run(0.0)
calib = float(np.mean(ex1 + ex0)); n_cal = len(ex1) + len(ex0)
tol = ALPHA + 2 * np.sqrt(ALPHA * (1 - ALPHA) / n_cal)          # nominal rate + 2 binomial SE
with tempfile.TemporaryDirectory() as d:
    recs = simulate(max(a.families, 3), a.dim, a.strength, seed=0)
    write(recs, d)
    back = load_recordings(d)
    r1, r2 = analyse_selection(recs, seed=0), analyse_selection(back, seed=0)
    roundtrip = abs(r1["test"][PRIMARY]["mean"] - r2["test"][PRIMARY]["mean"]) < 0.02 \
        and len(back["events"]) == len(recs["events"])

checks = [("planted effect detected (power >= 0.80)", power >= 0.80, f"power={power:.2f}, mean effect={m1:+.3f}"),
          ("position-bound null not detected (rate <= 0.05)", fpr <= 0.05, f"rate={fpr:.2f}, mean effect={m0:+.3f}"),
          (f"exact null rejected at about the nominal rate (<= {tol:.3f})", calib <= tol, f"rate={calib:.3f} over {n_cal} simulations"),
          ("on-disk format round-trips", roundtrip, "")]
print(f"families={a.families} (held-out families={nfam})  sims={a.sims}  strength={a.strength}")
for name, ok, info in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {info}")
if nfam < 6:
    print(f"  note: {nfam} held-out families cannot reach p <= 0.05 (smallest attainable p = {2.0 ** -nfam:.3f})")
sys.exit(0 if all(ok for _, ok, _ in checks) else 1)
