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
    assert z["h_cand"].shape[:3] == (4, 4, n_layers) and z["h_after_candidates"].shape[:2] == (4, n_layers)
    assert z["dist"].shape == (4, 4) and abs(z["dist"].sum(1) - 1).max() < 1e-4 and int(z["true"]) == 0
    assert len(list(act.glob("ref__*.npz"))) == len(keep)
    assert len(list(act.glob("*__at__*__native.npz"))) == 4 * len(reg["records"])
    out = tmp_path / "analysis.json"
    run("04_analyse.py", ["--activations", str(act), "--format", "native", "--min-dom", "0.0", "--out", str(out)], monkeypatch)
    res = json.load(open(out))
    assert res["status"] == "DEVELOPMENT ONLY" and res["n_events"] == len(reg["records"])
    pout = tmp_path / "patch.json"
    run("06_patching.py", common + ["--cells", "1", "--min-effect", "-1", "--ranks", "1", "--n-random", "1",
                                    "--out", str(pout)], monkeypatch)
    pr = json.load(open(pout))
    assert pr["sanity"]["pass"] and "restore_minus_random" in pr["test"]


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
