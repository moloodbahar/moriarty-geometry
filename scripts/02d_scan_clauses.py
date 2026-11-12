"""Scan EVERY clause of every step for its effect on the recorded model.

Why: the hand-written registry has 5 events. The geometry and patching
questions only need a clause that moves a goal's probability; they do not
need the full MORIARTY capture/release trajectory shape. This scan finds
such clauses in existing episodes so the instrument can be developed on
dozens of events instead of five.

For each step: incremental clause addition and single-clause deletion, on
all four goals. A clause is kept when, for one goal, addition and deletion
agree (addition gain >= 0.15 and deletion drop >= 0.10: the single-observer
form of the MORIARTY strict criteria). At most one event per step. The
least-effective other clause of the step is the inert control; steps with a
single clause are skipped.

Output: a registry file whose replacement fields are EMPTY, which the other
scripts read as "contrast = deletion of the clause" (neutral.condition_texts).

Status of these events: exploratory. They are selected on effect size, which
inflates the effect; the selection uses the NATIVE format and all later
analysis the DECOUPLED format, so the analysed measurement is not the one
selected on. They are not the preregistered sample (DECISIONS.md §6.6).
"""
import argparse, json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from moriarty_geometry.prompts import build_native
from moriarty_geometry.probe import OpenWeightsProbe, run_design
from moriarty_geometry.clauses import split_spans, edit_step, prefix_incremental
from moriarty_geometry.episodes import load_episodes
from moriarty_geometry.neutral import NeutralRecord, save_registry

ADD_GAIN, DEL_DROP = 0.15, 0.10
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--episodes", nargs="+", default=["data/episodes_confirmatory_reconstructed.json", "data/episodes_exploratory.json"])
ap.add_argument("--out", default="results/scan")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
conf = load_episodes(args.episodes)
probe = OpenWeightsProbe(args.model, device=args.device)

def dist(steps, goals):
    p, _ = run_design(probe, [build_native(steps, goals, r) for r in range(4)], capture=())
    return np.array([p[g] for g in goals])

records, table = [], []
for eid, rec in conf.items():
    ep = rec["episode"]; goals = ep["goals"]
    for step_no, step in enumerate(ep["steps"], start=1):
        spans = split_spans(step); k = len(spans); before = ep["steps"][: step_no - 1]
        if k < 2:
            continue
        add = np.array([dist(before + ([prefix_incremental(step, spans, j)] if j else []), goals) for j in range(k + 1)])
        full = dist(before + [step], goals)
        dele = np.array([dist(before + [edit_step(step, spans, i, None)], goals) for i in range(k)])
        gains = np.diff(add, axis=0)                 # (k, 4): what adding clause i does to each goal
        drops = full[None, :] - dele                 # (k, 4): what deleting clause i takes away
        score = np.minimum(gains, drops)             # agreement of the two interventions
        i, g = np.unravel_index(np.argmax(score), score.shape)
        size = np.abs(gains).max(1) + np.abs(drops).max(1); size[i] = np.inf
        inert = int(np.argmin(size))
        ok = bool(gains[i, g] >= ADD_GAIN and drops[i, g] >= DEL_DROP)
        row = {"episode_id": eid, "step": step_no, "clause_index": int(i), "goal_index": int(g), "kept": ok,
               "add_gain": float(gains[i, g]), "del_drop": float(drops[i, g]), "target_is_true": bool(g == 0),
               "target_dominant_after": bool(np.argmax(full) == g), "p_target_after": float(full[g]),
               "inert_index": inert, "inert_size": float(np.abs(gains[inert]).max() + np.abs(drops[inert]).max())}
        table.append(row)
        if ok:
            kind = "release" if g == 0 else "capture"       # toward the true goal / toward a wrong goal
            records.append(NeutralRecord(
                event_id=f"{eid}_s{step_no}_{kind}", episode_id=eid, step=step_no, clause_index=int(i),
                trigger_clause=spans[i].text(step).strip(), target_goal=goals[g], trigger_type=kind,
                neutral_primary="", neutral_secondary="", inert_index=inert, inert_neutral="", character="",
                notes=f"clause scan on {args.model}; contrast = deletion; add_gain={gains[i, g]:.3f} del_drop={drops[i, g]:.3f}",
                status="DRAFT"))
            print(f"{records[-1].event_id:34s} gain={gains[i, g]:+.2f} drop={drops[i, g]:+.2f} "
                  f"dominant_after={row['target_dominant_after']}")
json.dump(table, open(out / "clause_scan.json", "w"), indent=1)
save_registry(str(out / "registry_scan.json"), records)
fams = {r.episode_id.rsplit("_g", 1)[0] for r in records}
n_cap = sum(r.trigger_type == "capture" for r in records)
print(f"\n{len(table)} steps scanned, {len(records)} events kept ({n_cap} toward a wrong goal, {len(records) - n_cap} toward "
      f"the true goal) in {len(fams)} families -> {out / 'registry_scan.json'}")
print("With the 50/20/30 family split you need >= 20 families with events for 6 held-out families.")
