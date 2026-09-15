"""Stage 1 behavioural pilot on one open-weight model.

For every episode and prefix t = 0..6: run the NATIVE design (4 rotations)
and, optionally, the DECOUPLED design (16 cells), record the averaged goal
distribution, per-call coverage, rotation-specific argmaxes, and the
trajectory events. Then, for each event in the neutral registry, run the
present / neutral_primary / neutral_secondary / inert / deleted conditions
at the event's prefix.

Two outcomes are written separately, as they answer different questions:
  * clause_effect_transfer.json — Δp on the target goal, present vs neutral
  * trajectory_transfer.json    — events on the receiving model vs reference
"""
import argparse, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from moriarty_geometry.prompts import build_native, build_decoupled, decoupled_design, chars_block_level2
from moriarty_geometry.probe import OpenWeightsProbe, run_design
from moriarty_geometry.trajectory import make_point, events
from moriarty_geometry.clauses import split_spans, edit_step
from moriarty_geometry.neutral import load_registry

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--episodes", default="data/episodes_confirmatory_reconstructed.json")
ap.add_argument("--registry", default="data/neutral_registry.json")
ap.add_argument("--format", choices=["native", "decoupled", "both"], default="native")
ap.add_argument("--observer", choices=["naive", "agent_b"], default="naive")
ap.add_argument("--out", default="results/pilot")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

conf = json.load(open(args.episodes))
probe = OpenWeightsProbe(args.model, device=args.device)

def designs(steps, goals, cb):
    d = {}
    if args.format in ("native", "both"):
        d["native"] = [build_native(steps, goals, r, cb) for r in range(4)]
    if args.format in ("decoupled", "both"):
        d["decoupled"] = [build_decoupled(steps, goals, o, m, cb) for o, m in decoupled_design()]
    return d

traj_out, clause_out = {}, []
for eid, rec in conf.items():
    ep = rec["episode"]; goals = ep["goals"]; true = ep["true"]
    cb = None
    if args.observer == "agent_b":
        # Level-2 block: the least inferable core visible — the pipeline's
        # core report decides this; here the first cast member is shown as a
        # placeholder and must be replaced by the report's choice.
        cb = chars_block_level2([{"name": c["n"], "moral_core": c["c"]} for c in ep["cast"]], [ep["cast"][0]["n"]])
    traj_out[eid] = {}
    for fmt in designs([], goals, cb):
        pts = []; prev = None
        for t in range(0, len(ep["steps"]) + 1):
            specs = designs(ep["steps"][:t], goals, cb)[fmt]
            p, calls = run_design(probe, specs, capture=())
            pt = make_point(t, p, true, [c.argmax_goal for c in calls], [c.coverage for c in calls], prev)
            pts.append(pt); prev = pt
        traj_out[eid][fmt] = {"points": [vars(p) for p in pts], "events": events(pts)}
        print(eid, fmt, "q:", [round(p.q, 2) for p in pts], events(pts))

registry = load_registry(args.registry)
for r in registry:
    ep = conf[r.episode_id]["episode"]; goals = ep["goals"]; step = ep["steps"][r.step - 1]
    spans = split_spans(step)
    conds = {
        "present": step,
        "neutral_primary": edit_step(step, spans, r.clause_index, r.neutral_primary),
        "neutral_secondary": edit_step(step, spans, r.clause_index, r.neutral_secondary),
        "inert_neutral": edit_step(step, spans, r.inert_index, r.inert_neutral),
        "deleted": edit_step(step, spans, r.clause_index, None),
    }
    assert conds["present"] == step
    row = {"event_id": r.event_id, "target_goal": r.target_goal, "type": r.trigger_type}
    for name, text in conds.items():
        steps = ep["steps"][: r.step - 1] + [text]
        for fmt, specs in designs(steps, goals, None).items():
            p, calls = run_design(probe, specs, capture=())
            row[f"{fmt}:{name}"] = {"p_target": p[r.target_goal], "p": p, "min_coverage": min(c.coverage for c in calls)}
    for fmt in ("native", "decoupled"):
        if f"{fmt}:present" in row:
            row[f"{fmt}:delta_present_minus_neutral"] = row[f"{fmt}:present"]["p_target"] - row[f"{fmt}:neutral_primary"]["p_target"]
    clause_out.append(row)
    deltas = {k: round(v, 3) for k, v in row.items() if k.endswith("delta_present_minus_neutral")}
    print(r.event_id, deltas)

json.dump(traj_out, open(out / "trajectory_transfer.json", "w"), indent=1, default=str)
json.dump(clause_out, open(out / "clause_effect_transfer.json", "w"), indent=1)
print("wrote", out)
