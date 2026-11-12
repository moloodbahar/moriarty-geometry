import subprocess, sys, pathlib, json
import numpy as np
from moriarty_geometry.analysis.simulate import simulate, write
from moriarty_geometry.analysis.selection import (analyse_selection, family_split, cluster_bootstrap, family_signflip,
                                                    unique_order_cells, load_recordings, PRIMARY,
                                                    value_probe_and_truth_transfer)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_family_split_is_disjoint_and_needs_three():
    tr, dv, te = family_split([f"f{i}" for i in range(10)], seed=1)
    assert not (set(tr) & set(dv)) and not (set(tr) & set(te)) and not (set(dv) & set(te))
    assert len(tr) + len(dv) + len(te) == 10 and min(len(tr), len(dv), len(te)) >= 1
    try:
        family_split(["a", "b"]); assert False
    except ValueError:
        pass


def test_cluster_bootstrap_is_wider_than_naive_when_families_differ():
    v = [1, 1, 1, 1, 0, 0, 0, 0]; c = ["a"] * 4 + ["b"] * 4
    r = cluster_bootstrap(v, c)
    assert r["ci90"][0] <= 0.0 + 1e-9 and r["ci90"][1] >= 1.0 - 1e-9     # two clusters: anything from 0 to 1
    assert cluster_bootstrap([1.0, 2.0], ["a", "a"])["ci90"] is None


def test_pre_mapping_cells_are_deduplicated():
    recs = simulate(3, 32, seed=0)
    assert len(unique_order_cells(recs["traj"][0]["order"])) == 4


def test_signflip_is_exact_and_needs_five_families_to_reach_005():
    assert family_signflip([1.0, 2.0], ["a", "a"]) is None
    v = [1.0] * 4; c = list("abcd")
    assert abs(family_signflip(v, c) - 1 / 16) < 1e-12          # 4 families: cannot reach 0.05
    assert abs(family_signflip([1.0] * 5, list("abcde")) - 1 / 32) < 1e-12
    assert family_signflip([1, -1, 1, -1, 1, -1], list("abcdef")) > 0.3
    # events are averaged within family first: many events in one family do not shrink p
    assert abs(family_signflip([1.0] * 40, ["a"] * 37 + ["b", "c", "d"]) - 1 / 16) < 1e-12


def test_planted_selected_code_is_detected_and_null_is_not():
    hit = [analyse_selection(simulate(24, 48, 1.0, seed=s), seed=s) for s in range(3)]
    assert all(r["g1_supported"] and r["positive_control_detected"] and r["exceeds_lexical_control"] for r in hit)
    assert all(r["selected_layer_index"] > 0 for r in hit)          # layer 0 carries no selected code
    null = [analyse_selection(simulate(24, 48, 0.0, seed=s), seed=s) for s in range(4)]
    assert not any(r["g1_supported"] for r in null)
    # the position-bound code alone still moves the SAME-position index: that is why it is not the test
    assert np.mean([r["test"]["same:present_minus_neutral_primary"]["mean"] for r in null]) > 0.05
    # too few held-out families: flagged, and never "supported"
    small = analyse_selection(simulate(12, 48, 1.0, seed=0), seed=0)
    assert small["n_test_families"] < 6 and not small["enough_families_to_test"] and not small["g1_supported"]


def test_truth_transfer_runs_and_reports_wrong_commitments():
    recs = simulate(8, 48, 1.0, seed=3)
    fams = sorted({r["family"] for r in recs["traj"]})
    out = value_probe_and_truth_transfer(recs, fams[:5], fams[5:], 1)
    assert out["dominance_balanced_accuracy"] > 0.8
    assert out["wrong_commitments"]["top_is_dominant_wrong_goal"] > 0.8    # simulated value code is truth-blind


def test_analyse_script_end_to_end(tmp_path):
    write(simulate(6, 48, 1.0, seed=1), tmp_path / "act")
    assert len(load_recordings(str(tmp_path / "act"))["events"]) == 12
    out = tmp_path / "analysis.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "04_analyse.py"), "--activations", str(tmp_path / "act"),
                    "--out", str(out)], check=True, capture_output=True)
    res = json.load(open(out))
    assert res["status"] == "DEVELOPMENT ONLY" and res["G1:post_list"]["test"][PRIMARY]["mean"] > 0
