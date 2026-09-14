"""Verify that clause spans reconstruct every step exactly, and report the
original splitter's failures on the pilot trigger entries. Run first."""
import json, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from moriarty_geometry.clauses import split_spans, reconstruct

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

def old_split(step):
    sents = re.split(r"(?<=[.!?])\s+", step.strip()); out = []
    for s in sents:
        if len(s) > 140:
            out.extend(x.strip() for x in re.split(r",\s+(?=and |but |while |as |so )", s) if x.strip())
        elif s.strip(): out.append(s.strip())
    return out

eps = {e["episode_id"]: e for e in json.load(open(DATA / "episodes_exploratory.json"))}
conf = json.load(open(DATA / "episodes_confirmatory_reconstructed.json"))
n = bad_new = bad_old = 0
for e in list(eps.values()) + [r["episode"] | {"episode_id": k} for k, r in conf.items()]:
    for s in e["steps"]:
        n += 1
        bad_new += reconstruct(s, split_spans(s)) != s
        bad_old += " ".join(old_split(s)) != s
print(f"steps: {n}   new splitter failures: {bad_new}   original splitter failures: {bad_old}")
trig = json.load(open(DATA / "triggers_pilot.json"))
for x in trig["validated"] + trig["candidates"]:
    s = eps[x["episode_id"]]["steps"][x["step"] - 1]
    tag = "validated" if x in trig["validated"] else "candidate"
    print(f"  {x['episode_id']:22s} step {x['step']} {tag:9s} original-splitter exact: {' '.join(old_split(s)) == s}")
