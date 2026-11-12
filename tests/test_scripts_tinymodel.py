"""Run the recording, analysis and patching scripts end to end on a random
tiny model. The numbers mean nothing; the test is that every code path
executes on real prompts and that the files have the documented shapes."""
import json, pathlib, runpy, sys
import numpy as np
import pytest

pytest.importorskip("torch"); pytest.importorskip("transformers"); pytest.importorskip("tokenizers")
from test_probe_tinymodel import TinyProbe
import moriarty_geometry.probe as P

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(script, argv, monkeypatch, ok=(0, None)):
    monkeypatch.setattr(P, "OpenWeightsProbe", lambda *a, **k: TinyProbe())
    monkeypatch.setattr(sys, "argv", [script] + argv)
    try:
        runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")
    except SystemExit as e:
        assert e.code in ok, e.code


def small_episode_file(tmp_path):
    conf = json.load(open(ROOT / "data" / "episodes_confirmatory_reconstructed.json"))
    reg = json.load(open(ROOT / "data" / "neutral_registry.json"))
    keep = {r["episode_id"] for r in reg["records"]}
    eps = tmp_path / "eps.json"; json.dump({k: conf[k] for k in keep}, open(eps, "w"))
    return eps, reg, keep


def test_record_analyse_patch(tmp_path, monkeypatch):
    eps, reg, keep = small_episode_file(tmp_path)
    act = tmp_path / "act"
    common = ["--episodes", str(eps), "--device", "cpu"]
    run("03_record_activations.py", common + ["--format", "native", "--mode", "events", "--out", str(act)], monkeypatch)
    z = np.load(next(act.glob("*__at__present__native.npz")), allow_pickle=True)
    n_layers = len(z["layers"])
    assert z["h_cand"].shape[:3] == (4, 4, n_layers) and z["h_post_list"].shape[:2] == (4, n_layers)
    assert z["dist"].shape == (4, 4) and abs(z["dist"].sum(1) - 1).max() < 1e-4 and int(z["true"]) == 0
    assert len(list(act.glob("ref__*.npz"))) == len(keep)
    assert len(list(act.glob("*__at__*__native.npz"))) == 5 * len(reg["records"])      # 4 clause conditions + cue
    out = tmp_path / "analysis.json"
    run("04_analyse.py", ["--activations", str(act), "--format", "native", "--min-dom", "0.0", "--out", str(out)], monkeypatch)
    res = json.load(open(out))
    assert res["status"] == "DEVELOPMENT ONLY" and res["n_events"] == len(reg["records"])
    pout = tmp_path / "patch.json"
    run("06_patching.py", common + ["--pairs", "1", "--min-effect", "-1", "--ranks", "1", "--n-random", "1",
                                    "--blocks", "1", "2", "--out", str(pout)], monkeypatch)
    pr = json.load(open(pout))
    assert pr["sanity"]["pass"] and "restore_minus_random" in pr["subspace"]["test"]
    cc = pr["cross_cell"]["per_event"]
    assert set(cc) == {"order@post_list", "order@answer", "mapping@answer"}
    one = next(iter(cc["order@post_list"].values()))["1"]
    assert one["n_pairs"] == 1 and np.isfinite(one["target"]) and np.isfinite(one["competitor"])


def test_decoupled_recording_has_controls_and_noncyclic_orders(tmp_path, monkeypatch, capsys):
    eps, reg, keep = small_episode_file(tmp_path)
    act = tmp_path / "act"
    run("03_record_activations.py", ["--episodes", str(eps), "--device", "cpu", "--format", "decoupled",
                                     "--mode", "all", "--out", str(act)], monkeypatch)
    names = {p.name.split("__")[2] for p in act.glob("*__at__*__decoupled.npz")}
    assert names == {"present", "neutral_primary", "neutral_secondary", "inert_neutral", "cue", "ctl_present", "ctl_neutral_primary"}
    z = np.load(next(act.glob("*__at__present__decoupled.npz")), allow_pickle=True)
    assert z["h_cand"].shape[0] == 4 and len(z["cand_cells"]) == 4 and z["h_post_list"].shape[0] == 16
    orders = {tuple(int(i) for i in o) for o in z["order"]}
    assert orders == {(0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)}
    # read-outs before the mapping are identical across the four mappings of an order
    same = [c for c in range(16) if tuple(z["order"][c]) == (1, 0, 3, 2)]
    assert np.allclose(z["h_post_list"][same[0]], z["h_post_list"][same[1]])
    assert not np.allclose(z["h_answer"][same[0]], z["h_answer"][same[1]])
    t = np.load(next(act.glob("*__t3__traj__decoupled.npz")), allow_pickle=True)
    assert {tuple(int(i) for i in o) for o in t["order"]} != orders          # trajectory uses the cyclic orders
    out = tmp_path / "analysis.json"
    run("04_analyse.py", ["--activations", str(act), "--min-dom", "0.0", "--out", str(out)], monkeypatch)
    res = json.load(open(out)); g = res["G1:post_list"]
    assert res["status"] == "DEVELOPMENT ONLY" and not g["enough_families_to_test"]
    assert g["test"]["cross:cue_minus_neutral_primary"]["n_events"] >= 1
    assert g["test"]["cross:trigger_minus_lexical_control"]["n_events"] >= 1
    assert res["G2a:candidate_tokens"]["n_train"] > 0
    assert "cannot reach p <= 0.05" in capsys.readouterr().out


def test_clause_scan_writes_a_deletion_contrast_registry(tmp_path, monkeypatch):
    eps, reg, keep = small_episode_file(tmp_path)
    conf = json.load(open(eps)); k = sorted(conf)[0]
    conf = {k: conf[k]}; conf[k]["episode"]["steps"] = conf[k]["episode"]["steps"][:2]
    json.dump(conf, open(eps, "w"))
    out = tmp_path / "scan"
    run("02d_scan_clauses.py", ["--episodes", str(eps), "--device", "cpu", "--out", str(out)], monkeypatch)
    table = json.load(open(out / "clause_scan.json")); regs = json.load(open(out / "registry_scan.json"))
    assert len(table) >= 1 and all(r["neutral_primary"] == "" for r in regs["records"])



def test_pilot_smoke_report_and_localisation(tmp_path, monkeypatch, capsys):
    eps, reg, keep = small_episode_file(tmp_path)
    pilot = tmp_path / "pilot"
    run("02_behavioural_pilot.py", ["--episodes", str(eps), "--device", "cpu", "--format", "native", "--out", str(pilot)], monkeypatch)
    run("02b_smoke_report.py", [str(pilot)], monkeypatch, ok=(0, 1, None))     # a random model may fail the gates
    assert "label coverage" in capsys.readouterr().out
    loc = pilot / "localised_events.json"
    run("02c_localise_events.py", ["--episodes", str(eps), "--device", "cpu", "--pilot", str(pilot), "--out", str(loc)], monkeypatch)
    rows = json.load(open(loc))
    assert all(r["neutral_primary"] == "" and r["status"] == "DRAFT" for r in rows)


def test_deletion_contrast_and_cue_texts():
    from moriarty_geometry.neutral import NeutralRecord, condition_texts
    step = "Ana opened the door. Ben waited outside, and he checked his watch twice. Cy said nothing."
    r = NeutralRecord("e", "ep", 1, 1, "", "the goal", "capture", "", "", 0, "", "")
    t = condition_texts(r, step)
    assert "Ben waited" not in t["neutral_primary"] and "Ana opened" in t["neutral_primary"]
    assert "Ana opened" not in t["inert_neutral"] and "neutral_secondary" not in t
    assert t["cue"].startswith(t["neutral_primary"]) and t["cue"].endswith("the goal]")
