from moriarty_geometry.trajectory import make_point, events

def test_capture_and_release_detected():
    T = "true"; W = "wrong"
    seq = [
        {"true": .8, "wrong": .1, "x": .05, "y": .05},
        {"true": .5, "wrong": .3, "x": .1, "y": .1},
        {"true": .02, "wrong": .95, "x": .02, "y": .01},
        {"true": .01, "wrong": .97, "x": .01, "y": .01},
        {"true": .995, "wrong": .003, "x": .001, "y": .001},
    ]
    pts = []; prev = None
    for t, p in enumerate(seq):
        pt = make_point(t, p, T, [max(p, key=p.get)] * 4, [0.99] * 4, prev); pts.append(pt); prev = pt
    ev = events(pts)
    assert ev["wrong_entry"] == 2 and ev["wrong_collapse"] == 2
    assert ev["committed_wrong"] == [(2, 3)]
    assert ev["resolution"] == 4
