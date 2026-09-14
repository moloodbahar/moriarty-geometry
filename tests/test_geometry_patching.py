import numpy as np
from moriarty_geometry.analysis.geometry import ccgp, parallelism_score, cross_position_generalisation, shuffle_null
from moriarty_geometry.analysis.patching import learn_subspace, patch_restore, patch_remove, matched_random_patch, summarise, PatchOutcome

def synth(n_per=40, d=30, seed=0, abstract=True):
    rng = np.random.default_rng(seed)
    v_dich = rng.standard_normal(d); v_dich /= np.linalg.norm(v_dich)
    X, dich, cond = [], [], []
    for c in range(4):
        v_c = rng.standard_normal(d) * 0.5
        for dval in (0, 1):
            shift = v_dich * (2 if dval else -2)
            if not abstract:            # dichotomy direction differs per condition -> low CCGP
                shift = (rng.standard_normal(d)) * 2
            X.append(rng.standard_normal((n_per, d)) * 0.5 + v_c + shift)
            dich += [dval] * n_per; cond += [f"c{c}_{dval}"] * n_per
    return np.vstack(X), np.array(dich), np.array(cond)

def test_ccgp_high_for_abstract_low_for_entangled():
    X, y, c = synth(abstract=True); assert ccgp(X, y, c) > 0.9
    X, y, c = synth(abstract=False, seed=1); assert ccgp(X, y, c) < 0.8

def _pk(c):
    return np.array([x.split("_")[0] for x in c])

def test_parallelism_high_when_shared_direction():
    X, y, c = synth(abstract=True); assert parallelism_score(X, y, c, _pk(c)) > 0.8
    X, y, c = synth(abstract=False, seed=2); assert parallelism_score(X, y, c, _pk(c)) < 0.5

def test_cross_position_generalisation_recovers_content():
    rng = np.random.default_rng(0); d = 20
    content_vecs = rng.standard_normal((4, d))
    X, content, pos, fam = [], [], [], []
    for f in range(8):
        for g in range(4):
            for p in (0, 1):
                X.append(content_vecs[g] + rng.standard_normal(d) * 0.3); content.append(g); pos.append(p); fam.append(f)
    acc = cross_position_generalisation(np.array(X), np.array(content), np.array(pos), np.array(fam), 0, 1)
    assert acc > 0.9

def test_patching_restores_and_removes():
    rng = np.random.default_rng(0); d = 16
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    diffs = np.array([u * s + rng.standard_normal(d) * 0.05 for s in rng.uniform(1, 3, 12)])
    S = learn_subspace(diffs, rank=1, layer=5)
    assert abs(abs(S.U[:, 0] @ u) - 1) < 0.05
    hN = rng.standard_normal(d); hP = hN + 2 * u
    assert np.linalg.norm(patch_restore(hN, hP, S.U) - hP) < 0.2
    assert np.linalg.norm(patch_remove(hN, hP, S.U) - hN) < 0.2
    r = matched_random_patch(hN, hP, S.U, seed=1)
    assert abs(np.linalg.norm(r - hN) - np.linalg.norm(hP - hN)) < 0.2  # matched injection norm

def test_summarise_drops_low_coverage():
    out = [PatchOutcome("e1", "restore", .8, "w", .9), PatchOutcome("e1", "baseline_N", .2, "t", .9),
           PatchOutcome("e2", "restore", .7, "w", .3), PatchOutcome("e2", "baseline_N", .2, "t", .9)]
    s = summarise(out, "restore", "baseline_N")
    assert s["n"] == 1 and s["dropped_low_coverage"] == 1 and abs(s["mean_delta_p"] - .6) < 1e-9

def test_shuffle_null_gives_small_p_for_real_structure():
    X, y, c = synth(abstract=True, n_per=15)
    groups = _pk(c)   # permute within matched cells, across the dichotomy (as episode x step does in practice)
    obs, null, p = shuffle_null(lambda X_, y_: parallelism_score(X_, y_, c, _pk(c)), X, y, groups, n=50)
    assert p < 0.1
