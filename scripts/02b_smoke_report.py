"""Smoke tests on a behavioural pilot (output of 02_behavioural_pilot.py
--format both). Prints PASS/FAIL per check and exits non-zero if any gate
fails. Thresholds are the ones written in the test plan; change them there
first, not here.
"""
import argparse, json, sys, pathlib
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("pilot_dir"); a = ap.parse_args()
d = pathlib.Path(a.pilot_dir)
traj = json.load(open(d / "trajectory_transfer.json")); clause = json.load(open(d / "clause_effect_transfer.json"))
checks = []
def add(name, ok, info, gate=True): checks.append((name, ok, info, gate))

fmts = sorted({f for e in traj.values() for f in e})
for f in fmts:
    cov = [p["coverage_min"] for e in traj.values() for p in e[f]["points"]]
    add(f"[{f}] median label coverage >= 0.90", float(np.median(cov)) >= 0.90,
        f"median={np.median(cov):.3f} min={np.min(cov):.3f} share<0.50={np.mean(np.array(cov) < 0.5):.2f}")
    prior = [max(e[f]["points"][0]["p"].values()) for e in traj.values()]
    add(f"[{f}] no-story prior: no goal above 0.75", max(prior) <= 0.75,
        f"max over episodes={max(prior):.3f}, episodes above: {sum(x > 0.75 for x in prior)}/{len(prior)}")
    agree = [p["argmax_agreement"] for e in traj.values() for p in e[f]["points"][1:]]
    add(f"[{f}] cells agree on the dominant goal", float(np.mean(agree)) >= (3 if f == "native" else 12),
        f"mean agreeing cells={np.mean(agree):.1f} of {4 if f == 'native' else 16}", gate=False)
    ev = [e[f]["events"] for e in traj.values()]
    n_wrong = sum(1 for x in ev if x["wrong_entry"] is not None or x["wrong_collapse"] is not None)
    n_res = sum(1 for x in ev if x["resolution"] is not None); n_cw = sum(1 for x in ev if x["committed_wrong"])
    add(f"[{f}] events exist on this model", n_wrong + n_res > 0,
        f"wrong-dominant episodes={n_wrong}, resolutions={n_res}, committed-wrong runs={n_cw}, of {len(ev)} episodes")
if {"native", "decoupled"} <= set(fmts):
    qn = [p["q"] for e in traj.values() for p in e["native"]["points"]]
    qd = [p["q"] for e in traj.values() for p in e["decoupled"]["points"]]
    r = float(np.corrcoef(qn, qd)[0, 1])
    add("native and decoupled q-trajectories correlate (r >= 0.80)", r >= 0.80, f"r={r:.3f}")

for row in clause:
    eid = row["event_id"]
    for f in ("native", "decoupled"):
        if f"{f}:present" not in row: continue
        P = row[f"{f}:present"]["p_target"]; N1 = row[f"{f}:neutral_primary"]["p_target"]
        I = row[f"{f}:inert_neutral"]["p_target"]
        add(f"[{f}] {eid}: clause effect >= 0.15", P - N1 >= 0.15, f"present-neutral={P - N1:+.3f}", gate=False)
        add(f"[{f}] {eid}: inert edit moves p by < 0.05", abs(P - I) < 0.05, f"present-inert={P - I:+.3f}")
        if f"{f}:neutral_secondary" in row:
            N2 = row[f"{f}:neutral_secondary"]["p_target"]
            add(f"[{f}] {eid}: two neutral wordings agree within 0.10", abs(N1 - N2) < 0.10, f"N1-N2={N1 - N2:+.3f}")
        if f"{f}:cue" in row:
            C = row[f"{f}:cue"]["p_target"]
            add(f"[{f}] {eid}: explicit cue puts the target goal at >= 0.80", C >= 0.80, f"p_target with cue={C:.3f}")
    if "native:present" in row and "decoupled:present" in row:
        en = row["native:delta_present_minus_neutral"]; ed = row["decoupled:delta_present_minus_neutral"]
        if en >= 0.15:
            add(f"{eid}: decoupled effect >= half of native (prereg §3.4)", ed >= 0.5 * en, f"native={en:+.3f} decoupled={ed:+.3f}")
n_transfer = sum(1 for r in clause if r.get("native:delta_present_minus_neutral", r.get("decoupled:delta_present_minus_neutral", 0)) >= 0.15)
add("at least one registry clause transfers (>= 0.15)", n_transfer >= 1, f"{n_transfer}/{len(clause)} events")

for name, ok, info, gate in checks:
    print(f"[{'PASS' if ok else ('FAIL' if gate else 'note')}] {name}  ({info})")
failed = [c for c in checks if c[3] and not c[1]]
print(f"\n{len(failed)} gating check(s) failed" if failed else "\nall gating checks passed")
sys.exit(1 if failed else 0)
