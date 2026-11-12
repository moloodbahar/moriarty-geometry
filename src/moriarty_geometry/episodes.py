"""Load episodes from either shipped schema into one shape:

    {episode_id: {"episode": {"world", "cast": [{"n", "c"}], "true", "goals", "steps"}}}

goals[0] is always the true goal (canonical order); nothing downstream may
use the goal index as content across families.

`reports`: MORIARTY checks reports (results/<run>/checks_report.json). An
episode that appears in a report is kept only if that report marks it
usable. Episodes no report mentions are kept: the shipped confirmatory file
is already the usable set, and the shipped exploratory file has no report in
this repository (it contains unusable episodes; fine for development, not
for a confirmatory run).
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Sequence, Union


def load_episodes(paths: Union[str, Sequence[str]], reports: Optional[Sequence[str]] = None) -> Dict[str, dict]:
    if isinstance(paths, str):
        paths = [paths]
    out: Dict[str, dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):                       # confirmatory, reconstructed from call logs
            for k, v in d.items():
                out[k] = v if "episode" in v else {"episode": v}
        else:                                         # exploratory list (Datasets 1-2)
            for e in d:
                meta = e.get("meta", {})
                out[e["episode_id"]] = {"episode": {
                    "world": meta.get("world", ""),
                    "cast": [{"n": c["name"], "c": c["moral_core"]} for c in e.get("characters", [])],
                    "true": e["hidden_goal"],
                    "goals": [e["hidden_goal"]] + list(meta.get("distractor_goals", [])),
                    "steps": e["steps"]}}
    verdict: Dict[str, bool] = {}
    for path in reports or []:
        with open(path, encoding="utf-8") as f:
            for r in json.load(f)["results"]:
                verdict[r["episode_id"]] = bool(r["usable"])
    dropped = sorted(k for k in out if verdict.get(k) is False)
    if reports:
        unseen = sum(1 for k in out if k not in verdict)
        print(f"[episodes] {len(out)} loaded; {len(dropped)} dropped as not usable; "
              f"{unseen} not covered by any report (kept)", flush=True)
    for k in dropped:
        del out[k]
    for k, v in out.items():
        ep = v["episode"]
        assert len(ep["goals"]) == 4 and ep["goals"][0] == ep["true"], f"{k}: expected four goals with the true goal first"
    return out
