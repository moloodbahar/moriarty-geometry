"""Stage 2 recording.

Three kinds of file are written to --out (all .npz, float16):

  ref__<episode>.npz                       context-free reference vector of each goal text
  <episode>__t<k>__traj__<fmt>.npz         every prefix t = 0..T of every episode
  <event>__<pre|at>__<condition>__<fmt>.npz   each registry event under present /
                                           neutral_primary / neutral_secondary / inert_neutral

Per prompt cell the read-outs are: the last token of each candidate BY LIST
POSITION (h_cand), the token after the candidate list and before the answer
mapping (h_after_candidates), and the answer position (h_answer).

Run with --show-markers first and read the printed tokens.
"""
import argparse, json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from moriarty_geometry.prompts import build_native, build_decoupled, decoupled_design
from moriarty_geometry.probe import OpenWeightsProbe
from moriarty_geometry.clauses import split_spans, edit_step
from moriarty_geometry.neutral import load_registry
from moriarty_geometry.analysis.selection import save_recording, family_of

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--episodes", default="data/episodes_confirmatory_reconstructed.json")
ap.add_argument("--registry", default="data/neutral_registry.json")
ap.add_argument("--format", choices=["native", "decoupled"], default="decoupled")
ap.add_argument("--mode", choices=["all", "events", "trajectory"], default="all")
ap.add_argument("--layer-step", type=int, default=1, help="keep every k-th layer (and always the last)")
ap.add_argument("--show-markers", action="store_true", help="print the token at every read-out point and exit")
ap.add_argument("--out", default="results/activations")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
conf = json.load(open(args.episodes))
probe = OpenWeightsProbe(args.model, device=args.device)
L = probe.n_layers + 1
layers = sorted(set(range(0, L, args.layer_step)) | {L - 1})
probe.capture_layers = layers
CAND = [f"cand_end_{k}" for k in range(4)]
MARKERS = CAND + ["after_candidates", "answer"]

def specs_for(steps, goals):
    if args.format == "native":
        return [build_native(steps, goals, r) for r in range(4)]
    return [build_decoupled(steps, goals, o, m) for o, m in decoupled_design()]

if args.show_markers:
    ep = next(iter(conf.values()))["episode"]
    for steps in ([], ep["steps"][:1]):
        spec = specs_for(steps, ep["goals"])[1]
        print(f"--- {args.format}, {len(steps)} step(s) ---")
        for name, (i, tok) in sorted(probe.describe_markers(spec).items(), key=lambda kv: kv[1][0]):
            print(f"  {name:18s} token {i:5d}  {tok!r}")
    print("Expected: cand_end_k = last token of candidate k's text (not a newline or the next number);\n"
          "after_candidates = cand_end_3; answer = the last token of the generation prompt.")
    sys.exit(0)

def record(steps, goals, path, **meta):
    dist, order, mapping, cov, H, HC = [], [], [], [], {"after_candidates": [], "answer": []}, []
    for spec in specs_for(steps, goals):
        res = probe.call(spec, capture=MARKERS)
        dist.append([res.goal_dist[g] for g in goals]); order.append(spec.order); mapping.append(spec.mapping)
        cov.append(res.coverage)
        for m in H: H[m].append(res.hidden[m].float().numpy())
        HC.append(np.stack([res.hidden[c].float().numpy() for c in CAND]))
    save_recording(path, goals=goals, dist=dist, order=order, mapping=mapping,
                   hidden={m: np.stack(v) for m, v in H.items()}, h_cand=np.stack(HC), layers=layers,
                   coverage=cov, **meta)

def goal_index(goals, text):
    if text in goals: return goals.index(text)
    hits = [i for i, g in enumerate(goals) if g.startswith(text[:40])]
    assert len(hits) == 1, f"goal not found uniquely: {text[:60]!r}"
    return hits[0]

for eid, rec in conf.items():
    ep = rec["episode"]; goals = ep["goals"]
    ref = out / f"ref__{eid}.npz"
    if not ref.exists():
        vecs = probe.reference_vectors(goals).numpy()[:, layers, :]
        np.savez_compressed(ref, episode=eid, goals=np.array(goals), vecs=vecs.astype(np.float16))

if args.mode in ("all", "trajectory"):
    for eid, rec in conf.items():
        ep = rec["episode"]; goals = ep["goals"]
        for t in range(len(ep["steps"]) + 1):
            record(ep["steps"][:t], goals, out / f"{eid}__t{t}__traj__{args.format}.npz",
                   family=family_of(eid), episode=eid, true=goal_index(goals, ep["true"]))
        print("trajectory", eid)

if args.mode in ("all", "events"):
    registry = load_registry(args.registry)
    if any(r.status != "FROZEN" for r in registry):
        print("NOTE: neutral registry is not FROZEN — these recordings are development data only.")
    for r in registry:
        ep = conf[r.episode_id]["episode"]; goals = ep["goals"]; step = ep["steps"][r.step - 1]
        spans = split_spans(step); before = ep["steps"][: r.step - 1]
        meta = dict(family=family_of(r.episode_id), episode=r.episode_id, kind=r.trigger_type,
                    target=goal_index(goals, r.target_goal), true=goal_index(goals, ep["true"]))
        conds = {"present": step,
                 "neutral_primary": edit_step(step, spans, r.clause_index, r.neutral_primary),
                 "neutral_secondary": edit_step(step, spans, r.clause_index, r.neutral_secondary),
                 "inert_neutral": edit_step(step, spans, r.inert_index, r.inert_neutral)}
        record(before, goals, out / f"{r.event_id}__pre__present__{args.format}.npz", **meta)
        for cname, text in conds.items():
            record(before + [text], goals, out / f"{r.event_id}__at__{cname}__{args.format}.npz", **meta)
        print("event", r.event_id)
