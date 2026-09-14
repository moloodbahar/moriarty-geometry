"""Count yields per family from a pilot run and estimate the generated
sample needed for a target number of strict captures (keeps the strict and
joint denominators separate)."""
import argparse, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from moriarty_geometry.analysis.feasibility import FamilyYield, counts, episodes_needed
ap = argparse.ArgumentParser(); ap.add_argument("yields_json"); ap.add_argument("--target", type=int, default=25)
a = ap.parse_args()
fams = [FamilyYield(**r) for r in json.load(open(a.yields_json))]
print(counts(fams))
for key in ("strict_captures", "joint_events", "two_clause_captures"):
    print(key, episodes_needed(fams, a.target, key))
