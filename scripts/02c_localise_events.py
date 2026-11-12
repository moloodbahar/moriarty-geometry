"""Clause localisation ON THE RECORDED MODEL.

For every wrong-entry / wrong-collapse / resolution step that
02_behavioural_pilot.py detected on this model, run incremental clause
addition and single-clause deletion and apply the MORIARTY strict criteria
for one observer (same clause under addition and deletion; addition gain
>= 0.15; deletion drop >= 0.10). Writes candidate registry records with the
neutral fields EMPTY: the replacements are written by hand, judge-reviewed
and frozen before any activation is recorded for them.

Whether these events or the transferred gpt-4o-mini triggers define the
geometry sample is an open decision (DECISIONS.md §5.6).
"""
import argparse, json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from moriarty_geometry.prompts import build_native
from moriarty_geometry.probe import OpenWeightsProbe, run_design
from moriarty_geometry.episodes import load_episodes
from moriarty_geometry.clauses import split_spans, edit_step, prefix_incremental

ADD_GAIN, DEL_DROP = 0.15, 0.10
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--episodes", nargs="+", default=["data/episodes_confirmatory_reconstructed.json"])
ap.add_argument("--report", nargs="*", default=None, help="MORIARTY checks_report.json files; episodes they mark not usable are dropped")
ap.add_argument("--pilot", default="results/pilot", help="directory holding trajectory_transfer.json")
ap.add_argument("--out", default="results/pilot/localised_events.json")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
conf = load_episodes(args.episodes, args.report); traj = json.load(open(pathlib.Path(args.pilot) / "trajectory_transfer.json"))
probe = OpenWeightsProbe(args.model, device=args.device)

def p_goal(steps, goals, goal):
    p, _ = run_design(probe, [build_native(steps, goals, r) for r in range(4)], capture=())
    return p[goal]

found = []
for eid, rec in conf.items():
    ep = rec["episode"]; goals = ep["goals"]; t = traj[eid]["native"]; pts = t["points"]; ev = t["events"]
    todo = {}
    for kind in ("wrong_entry", "wrong_collapse"):
        if ev[kind] is not None: todo.setdefault(int(ev[kind]), ("capture", pts[int(ev[kind])]["dominant"]))
    if ev["resolution"] is not None: todo.setdefault(int(ev["resolution"]), ("release", ep["true"]))
    for step_no, (kind, target) in sorted(todo.items()):
        step = ep["steps"][step_no - 1]; spans = split_spans(step); before = ep["steps"][: step_no - 1]
        if len(spans) < 2: continue
        add = [p_goal(before + ([prefix_incremental(step, spans, k)] if k else []), goals, target) for k in range(len(spans) + 1)]
        gains = np.diff(add)
        full = p_goal(before + [step], goals, target)
        drops = np.array([full - p_goal(before + [edit_step(step, spans, i, None)], goals, target) for i in range(len(spans))])
        ia, idl = int(np.argmax(gains)), int(np.argmax(drops))
        strict = bool(ia == idl and gains[ia] >= ADD_GAIN and drops[idl] >= DEL_DROP)
        two_clause = bool(not strict and abs(ia - idl) == 1 and gains[ia] >= ADD_GAIN and drops[idl] >= DEL_DROP)
        inert = int(np.argmin(np.abs(gains) + np.abs(drops) + 1e3 * (np.arange(len(spans)) == ia)))
        row = {"event_id": f"{eid}_s{step_no}_{kind}", "episode_id": eid, "step": step_no, "clause_index": ia,
               "trigger_clause": spans[ia].text(step).strip(), "target_goal": target, "trigger_type": kind,
               "neutral_primary": "", "neutral_secondary": "", "inert_index": inert, "inert_neutral": "",
               "character": "", "notes": f"localised on {args.model}", "status": "DRAFT",
               "_strict": strict, "_two_clause": two_clause, "_add_gains": [float(x) for x in gains],
               "_del_drops": [float(x) for x in drops]}
        found.append(row)
        print(f"{row['event_id']:34s} {'STRICT' if strict else 'two-clause' if two_clause else 'not localised':13s} "
              f"clause {ia} gain={gains[ia]:+.2f} drop={drops[idl]:+.2f}")
json.dump(found, open(args.out, "w"), indent=1, ensure_ascii=False)
n_s = sum(r["_strict"] for r in found)
print(f"\n{len(found)} event steps, {n_s} strict, {sum(r['_two_clause'] for r in found)} two-clause -> {args.out}")
