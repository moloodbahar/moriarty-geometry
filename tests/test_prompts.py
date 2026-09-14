from moriarty_geometry.prompts import (build_native, build_decoupled, rotations_native, latin_orders,
                                       latin_mappings, decoupled_design, LETTERS)

GOALS = ["true goal text", "distractor one", "distractor two", "distractor three"]
STEPS = ["Step one.", "Step two."]

def test_native_rotation_cycles_every_goal_through_every_letter():
    seen = {g: set() for g in GOALS}
    for r in range(4):
        spec = build_native(STEPS, GOALS, r)
        for L, g in spec.letter_to_goal.items():
            seen[g].add(L)
    assert all(v == set(LETTERS) for v in seen.values())

def test_native_marker_precedes_question():
    spec = build_native(STEPS, GOALS, 0)
    assert spec.user[spec.markers["after_candidates"]:].startswith("\n\nWhich is most likely")

def test_decoupled_crosses_position_and_letter():
    cells = decoupled_design()
    assert len(cells) == 16
    # every goal appears at every position and under every letter
    pos = {g: set() for g in GOALS}; let = {g: set() for g in GOALS}
    for o, m in cells:
        spec = build_decoupled(STEPS, GOALS, o, m)
        for p, gi in enumerate(o):
            pos[GOALS[gi]].add(p); let[GOALS[gi]].add(m[p])
        assert spec.markers["after_candidates"] < spec.markers["after_mapping"] < len(spec.user)
        assert "ANSWER MAPPING" not in spec.user[:spec.markers["after_candidates"]]
    assert all(v == {0, 1, 2, 3} for v in pos.values())
    assert all(v == set(LETTERS) for v in let.values())

def test_mappings_are_derangements_except_identity():
    ms = latin_mappings()
    assert ("A", "B", "C", "D") in ms
    for m in ms:
        if m == ("A", "B", "C", "D"): continue
        assert all(m[i] != LETTERS[i] for i in range(4))
