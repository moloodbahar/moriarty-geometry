"""Lint MORIARTY seed families before paying for the prior gate (Check 0).

Mechanical checks only. They catch the three failure modes the v4.0-v4.3
seed iterations ran into, in the crude form a script can see:

  premise leak        a content word of the WORLD occurs in exactly one goal
  core-goal echo      a content word of a character's moral core recurs in
                      the goal aimed at that character
  salience asymmetry  the four goals differ a lot in length, or do not share
                      the fixed shape (three "Get <Name> to ..., believing ..."
                      goals, one per character, and one "Lead the ... each
                      believing ..." group goal)

plus schema, duplicate ids and names reused from earlier families.

Passing this lint does NOT mean a family passes Check 0: only the measured
pick shares decide that (max share <= 0.75, no dead goal). The seed file
shipped as data/seeds_v5_draft.json has passed this lint and nothing else.

    python scripts/lint_seed_families.py data/seeds_v5_draft.json \
        --existing ../Moriarty/seeds_v4_confirmatory.json ../Moriarty/seeds_v3_1.json
"""
import argparse, json, re, sys

STOP = set("""the a an and or of to in on at for with from by is are was be been it its his her their this that these
those as not no nor never ever any every each one two three all still before after than then when who whom whose what
which while into over under about them they she he own self herself himself whatever however anyone anything nothing
something someone everyone others other another same more most much many few several some only even also just very
get lead believing believe week weeks month months season ordinary normal small three""".split())

def stems(text):
    out = set()
    for w in re.findall(r"[a-zA-Z]+", text.lower()):
        if len(w) < 4 or w in STOP:
            continue
        for suf in ("ings", "ing", "edly", "ed", "es", "s", "ly"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        out.add(w)
    return out

def lint_family(f, taken_ids, taken_names):
    errs, warns = [], []
    fid = f.get("family_id", "?")
    if fid in taken_ids: errs.append("duplicate family_id")
    chars, goals = f.get("characters", []), f.get("goal_family", [])
    if len(chars) != 3 or len(goals) != 4:
        return errs + ["need exactly 3 characters and 4 goals"], warns
    names = [c["name"] for c in chars]
    for n in names:
        if n in taken_names: errs.append(f"name reused from another family: {n}")
    targeted = []
    for g in goals[:3]:
        m = re.match(r"Get (\S+) to ", g)
        hit = [n for n in names if re.search(rf"\b{re.escape(n)}\b", g)]
        if not m or m.group(1) not in names or len(hit) != 1: errs.append(f"individual goal must name exactly one character: {g[:50]}")
        else: targeted.append(m.group(1))
        if ", believing " not in g and " at exactly the moment " not in g and " at the moment " not in g:
            warns.append(f"no false-belief clause: {g[:50]}")
    if sorted(targeted) != sorted(names): errs.append("each character must be the target of exactly one goal")
    g4 = goals[3]
    if not g4.startswith("Lead the ") or "each believing" not in g4: errs.append("goal 4 must be 'Lead the ..., each believing ...'")
    if any(re.search(rf"\b{re.escape(n)}\b", g4) for n in names): errs.append("group goal names a character")
    lens = [len(g.split()) for g in goals]
    if max(lens) / min(lens) > 1.6: warns.append(f"goal lengths uneven (words: {lens})")
    for c in chars:
        own = next((g for g in goals[:3] if g.startswith(f"Get {c['name']} to ")), "")
        echo = (stems(c["moral_core"]) & stems(own)) - stems(c["name"])
        if echo: warns.append(f"core-goal echo for {c['name']}: {sorted(echo)}")
    world = stems(f.get("world", ""))
    gs = [stems(g) for g in goals]
    m = re.match(r"Lead the (\w+)", g4)             # the group noun is part of the fixed shape of goal 4
    gs[3] = gs[3] - (stems(m.group(1)) if m else set())
    for w in sorted(world):
        where = [i + 1 for i, s in enumerate(gs) if w in s]
        if len(where) == 1: warns.append(f"premise leak: world word '{w}' occurs only in goal {where[0]}")
    return errs, warns

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds")
    ap.add_argument("--existing", nargs="*", default=[], help="earlier seed files; their ids and names must not be reused")
    a = ap.parse_args()
    taken_ids, taken_names = set(), set()
    for p in a.existing:
        for f in json.load(open(p, encoding="utf-8")):
            taken_ids.add(f.get("family_id") or f.get("seed_id") or f.get("episode_id"))
            taken_names |= {c["name"] for c in f.get("characters", [])}
    n_err = n_warn = 0
    fams = json.load(open(a.seeds, encoding="utf-8"))
    for f in fams:
        errs, warns = lint_family(f, taken_ids, taken_names)
        taken_ids.add(f.get("family_id")); taken_names |= {c["name"] for c in f.get("characters", [])}
        n_err += len(errs); n_warn += len(warns)
        for e in errs: print(f"ERROR {f.get('family_id')}: {e}")
        for w in warns: print(f"warn  {f.get('family_id')}: {w}")
    print(f"{len(fams)} families, {n_err} errors, {n_warn} warnings")
    sys.exit(1 if n_err else 0)
