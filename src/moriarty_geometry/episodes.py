"""Load episodes from either shipped schema into one shape:

    {episode_id: {"episode": {"world", "cast": [{"n", "c"}], "true", "goals", "steps"}}}

goals[0] is always the true goal (canonical order); nothing downstream may
use the goal index as content across families.
"""
from __future__ import annotations

import json
from typing import Dict, Sequence, Union


def load_episodes(paths: Union[str, Sequence[str]]) -> Dict[str, dict]:
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
    for k, v in out.items():
        ep = v["episode"]
        assert len(ep["goals"]) == 4 and ep["goals"][0] == ep["true"], f"{k}: expected four goals with the true goal first"
    return out
