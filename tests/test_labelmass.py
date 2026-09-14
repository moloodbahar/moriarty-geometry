import numpy as np
from moriarty_geometry.probe import label_mass, to_goal_distribution, average_distributions
from moriarty_geometry.prompts import build_native

def test_label_mass_merges_forms_and_reports_coverage():
    probs = np.zeros(20); probs[1] = 0.3; probs[2] = 0.1; probs[5] = 0.2; probs[9] = 0.4  # last is junk
    sets = {"A": [1, 2], "B": [5], "C": [], "D": []}
    lm, cov = label_mass(probs, sets)
    assert abs(cov - 0.6) < 1e-9
    assert abs(lm["A"] - 0.4/0.6) < 1e-9 and abs(lm["B"] - 0.2/0.6) < 1e-9 and lm["C"] == 0.0

def test_average_over_rotations_maps_back_to_goals():
    goals = ["g0", "g1", "g2", "g3"]
    dists = []
    for r in range(4):
        spec = build_native([], goals, r)
        lm = {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1}   # pure position bias toward A
        dists.append(to_goal_distribution(lm, spec))
    avg = average_distributions(dists)
    assert all(abs(v - 0.25) < 1e-9 for v in avg.values())  # rotation removes pure letter bias
