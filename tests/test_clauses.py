import json, re, pathlib
from moriarty_geometry.clauses import split_spans, reconstruct, clause_texts, edit_step, prefix_incremental, find_clause

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

def all_steps():
    for e in json.load(open(DATA / "episodes_exploratory.json")):
        for s in e["steps"]:
            yield e["episode_id"], s

def test_reconstruction_is_exact_on_every_exploratory_step():
    for eid, s in all_steps():
        assert reconstruct(s, split_spans(s)) == s, eid

def test_boundaries_match_original_splitter():
    # the original rule, reproduced here, minus its comma loss
    def old(step):
        sents = re.split(r"(?<=[.!?])\s+", step.strip()); out = []
        for s in sents:
            if len(s) > 140:
                out.extend(x.strip() for x in re.split(r",\s+(?=and |but |while |as |so )", s) if x.strip())
            elif s.strip(): out.append(s.strip())
        return out
    for eid, s in all_steps():
        assert [c.rstrip(",") for c in old(s)] == [c.rstrip(",") for c in clause_texts(s)], eid

def test_edit_preserves_context_and_deletion_removes_clause():
    _, s = next(all_steps())
    sp = split_spans(s)
    if len(sp) < 2: return
    rep = edit_step(s, sp, 1, "NEUTRAL SENTENCE.")
    assert rep.startswith(sp[0].text(s)) and "NEUTRAL SENTENCE." in rep
    dele = edit_step(s, sp, 1, None)
    assert sp[1].text(s).strip() not in dele
    assert dele.startswith(sp[0].text(s).rstrip())

def test_incremental_prefix_of_full_equals_step():
    _, s = next(all_steps())
    sp = split_spans(s)
    assert prefix_incremental(s, sp, len(sp)) == s.rstrip()

def test_find_clause_tolerates_truncation():
    _, s = next(all_steps())
    sp = split_spans(s)
    full = sp[-1].text(s).strip()
    assert find_clause(s, sp, full[:45]) == len(sp) - 1
