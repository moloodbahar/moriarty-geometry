"""Stage 2 recording. For each event and prefix, capture residual-stream
activations at the named markers under every condition in the design, and
save one .npz per (event, condition, format) with arrays of shape
(n_cells, n_layers, d) per marker plus the per-cell goal distributions."""
import argparse, json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from moriarty_geometry.prompts import build_native, build_decoupled, decoupled_design
from moriarty_geometry.probe import OpenWeightsProbe
from moriarty_geometry.clauses import split_spans, edit_step
from moriarty_geometry.neutral import load_registry

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--episodes", default="data/episodes_confirmatory_reconstructed.json")
ap.add_argument("--registry", default="data/neutral_registry.json")
ap.add_argument("--format", choices=["native", "decoupled"], default="decoupled")
ap.add_argument("--markers", nargs="+", default=["after_candidates", "answer"])
ap.add_argument("--out", default="results/activations")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
conf = json.load(open(args.episodes)); probe = OpenWeightsProbe(args.model, device=args.device)

def specs_for(steps, goals):
    if args.format == "native": return [build_native(steps, goals, r) for r in range(4)]
    return [build_decoupled(steps, goals, o, m) for o, m in decoupled_design()]

for r in load_registry(args.registry):
    ep = conf[r.episode_id]["episode"]; goals = ep["goals"]; step = ep["steps"][r.step - 1]; spans = split_spans(step)
    conds = {"present": step, "neutral_primary": edit_step(step, spans, r.clause_index, r.neutral_primary),
             "inert_neutral": edit_step(step, spans, r.inert_index, r.inert_neutral)}
    # also the pre-event prefix (t-1) and, for capture events, t+1..T for the committed run
    for t_off, label in ((-1, "pre"), (0, "at")):
        for cname, text in conds.items():
            if t_off == -1 and cname != "present": continue
            steps = ep["steps"][: r.step - 1] if t_off == -1 else ep["steps"][: r.step - 1] + [text]
            H = {m: [] for m in args.markers}; dists = []; orders = []; maps = []
            for spec in specs_for(steps, goals):
                res = probe.call(spec, capture=args.markers)
                for m in args.markers: H[m].append(res.hidden[m].float().numpy())
                dists.append([res.goal_dist[g] for g in goals]); orders.append(spec.order); maps.append(spec.mapping)
            np.savez_compressed(out / f"{r.event_id}__{label}__{cname}__{args.format}.npz",
                                goals=np.array(goals), dist=np.array(dists), order=np.array(orders), mapping=np.array(maps),
                                **{f"h_{m}": np.stack(H[m]) for m in args.markers})
            print("saved", r.event_id, label, cname)
