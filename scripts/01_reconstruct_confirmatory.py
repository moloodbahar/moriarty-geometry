"""Reconstruct confirmatory episodes and Agent-B posteriors from the run's
own call logs (episodes_calls.jsonl, gd_steps_agentB_calls.jsonl,
gd_clauses_*_calls.jsonl). Output: data/episodes_confirmatory_reconstructed.json
This is the script that produced the shipped file; rerun it against the
manuscript's exact confirmatory run directory to regenerate."""
import argparse, json, re, math, collections, pathlib
ap = argparse.ArgumentParser()
ap.add_argument("--logs", required=True, help="directory containing the *_calls.jsonl files")
ap.add_argument("--out", default="data/episodes_confirmatory_reconstructed.json")
args = ap.parse_args()
U = pathlib.Path(args.logs)
LABELS = ["A", "B", "C", "D"]

def parse_sys(s):
    world = re.search(r"WORLD: (.+)", s).group(1).strip()
    chars = re.findall(r"^- (.+?): moral core = (.+)$", s, re.M)
    goal = re.search(r"YOUR HIDDEN GOAL: (.+)", s).group(1).strip()
    dis = [l[2:].strip() for l in s.split("alongside your real one:")[1].split("Rules:")[0].strip().splitlines() if l.startswith("- ")]
    return world, chars, goal, dis

def story_so_far(u):
    block = u.split("STORY SO FAR:")[1].split("Steps remaining")[0].strip()
    if block.startswith("(") or not block: return []
    return [re.sub(r"^\[step \d+\]\s*", "", l).strip() for l in block.splitlines() if l.strip()]

calls = []
for l in open(U / "episodes_calls.jsonl"):
    r = json.loads(l)
    if not r.get("ok"): continue
    try: body = json.loads(r["response_raw"])
    except Exception: continue
    w, ch, g, d = parse_sys(r["system"])
    rem = int(re.search(r"Steps remaining including this one: (\d+)", r["user"]).group(1))
    calls.append(dict(w=w, ch=ch, g=g, d=d, rem=rem, prefix=story_so_far(r["user"]),
                      step=body.get("step", "").strip(), rat=body.get("private_rationale", "").strip()))
idx = {}
for c in calls: idx.setdefault((c["g"], tuple(c["prefix"]), c["step"]), c["rat"])
episodes = []
for c in calls:
    if c["rem"] != 1: continue
    steps = c["prefix"] + [c["step"]]
    if len(steps) != 6: continue
    rats = [idx.get((c["g"], tuple(steps[:k]), steps[k]), "") for k in range(6)]
    episodes.append(dict(world=c["w"], cast=[{"n": a, "c": b} for a, b in c["ch"]], true=c["g"],
                         goals=[c["g"]] + c["d"], steps=steps, rationales=rats))
by_first = {}
for e in episodes: by_first.setdefault(e["steps"][0][:90], e)

def p_labels(top):
    mass = collections.defaultdict(float)
    for tok, lp in top:
        t = tok.strip().upper()
        if t in LABELS: mass[t] += math.exp(lp)
    tot = sum(mass.values()); return {L: (mass.get(L, 0.0) / tot if tot else 0.0) for L in LABELS}
def goals_from_prompt(u):
    b = u.split("CANDIDATE HIDDEN GOALS:")[1]
    return {L: re.search(rf"^{L}\. (.+)$", b, re.M).group(1).strip() for L in LABELS}
def aggregate(cs, true_goal):
    per = collections.defaultdict(list); am = []; covs = []
    for r in cs:
        gm = goals_from_prompt(r["user"]); pl = p_labels(r["top_logprobs"]); covs.append(r.get("coverage", 1.0))
        pg = {gm[L]: pl[L] for L in gm}; am.append(max(pg, key=pg.get))
        for g, v in pg.items(): per[g].append(v)
    p = {g: sum(v) / len(v) for g, v in per.items()}
    dom = max(p, key=p.get)
    H = -sum(v * math.log2(v) for v in p.values() if v > 0) / 2
    return dict(p={g: round(v, 4) for g, v in p.items()}, q=round(p.get(true_goal, 0), 4),
                W=round(max(v for g, v in p.items() if g != true_goal), 4), H=round(H, 4),
                dominant_goal=dom, dominant_is_true=dom == true_goal,
                argmax_agreement=collections.Counter(am).get(dom, 0), min_coverage=round(min(covs), 4))
def first_step_of_prompt(u):
    m = re.search(r"\[step 1\] (.+?)(?=\n\[step 2\]|\n\n)", u, re.S); return m.group(1).strip() if m else None

steps_b = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(U / "gd_steps_agentB_calls.jsonl"):
    r = json.loads(l)
    if not r.get("ok"): continue
    m = re.match(r"steps_(.+)_t(\d+)_perm(\d)", r["purpose"])
    if m: steps_b[m.group(1)][int(m.group(2))].append(r)
out = {}
for eid, tmap in sorted(steps_b.items()):
    key = None
    for t in sorted(tmap):
        if t == 0: continue
        f = first_step_of_prompt(tmap[t][0]["user"])
        if f: key = f[:90]; break
    ep = by_first.get(key)
    if not ep: print("NO MATCH", eid); continue
    pts = []
    for t in sorted(tmap):
        d = aggregate(tmap[t], ep["true"]); d["t"] = t; pts.append(d)
    out[eid] = {"episode": ep, "agentB": pts}
json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
print("episodes matched:", len(out), "->", args.out)
