# moriarty-geometry

Behavioural transfer, representational geometry and paired activation
patching of *interpretive capture* on open-weight observers. This is the
stage-2/3 companion to the MORIARTY framework: it takes validated capture
and release positions and asks whether they have a linear and causal
counterpart inside a model whose activations can be read.

Read `DECISIONS.md` first — it records what was verified in the original
code, what was wrong with the earlier plan, and why this design was chosen.
`PREREGISTRATION.md` (v2, draft) holds the hypotheses, thresholds and
stopping rules; it is frozen only after the feasibility pilot. §10 lists
amendments that are implemented but not yet decided.

## What is fixed relative to the original repository

- **Clause edits are exact.** Clauses are character spans; every edit
  asserts the untouched text is byte-identical (`clauses.py`). The original
  splitter dropped commas, so the "present" condition differed from the
  measured text for 2 of 5 validated pilot triggers.
- **Logit lens handles the final norm once** and asserts the reconstructed
  final-layer logits equal the model's logits (`probe.py`). The original
  readout double-normalised the last layer.
- **Label mass is exact** over the full vocabulary, with coverage recorded.
- **Neutral replacements are per event**, character-preserving,
  length-matched, content-checked and hash-frozen (`neutral.py`,
  `data/neutral_registry.json` — status DRAFT until judge-reviewed).
- **A decoupled prompt format** crosses list position with answer letter
  and exposes a read-out point before the mapping (`prompts.py`).
- **Patching uses the event's own paired difference** through a frozen
  learned subspace, with matched random-subspace controls
  (`analysis/patching.py`). No universal "wrong-goal direction" is assumed.

## Layout

```
src/moriarty_geometry/
  clauses.py        span-based clause splitting and editing, exact reconstruction
  episodes.py       one loader for both shipped episode schemas
  prompts.py        NATIVE (4 rotations) and DECOUPLED (16 cells) probe formats; read-out markers;
                    control-task prompt; second (non-cyclic) set of list orders
  probe.py          open-weights probe: exact label mass, marker capture, logit lens,
                    block-level capture and patching, goal reference vectors
  trajectory.py     q, W, H, JSD and the five MORIARTY events (thresholds unchanged)
  neutral.py        per-event neutral-replacement registry with validation + hash;
                    text of every clause condition (replacement, deletion, explicit cue)
  analysis/
    selection.py    cross-position selection index (primary G1/H3 statistic), family splits,
                    family sign-flip test, positive and lexical controls, candidate-token probes
    simulate.py     synthetic recordings with a planted or absent selected code
    decoding.py     family-split dominance probe, layer selection, regime contrasts
    geometry.py     CCGP, parallelism, shuffle null
    patching.py     paired-difference subspace helpers
    feasibility.py  yields per denominator, family bootstrap for sample size
scripts/
  00_verify_reconstruction.py    run first
  00b_synthetic_validation.py    step 0: planted effect detected, null not detected
  01_reconstruct_confirmatory.py rebuild episodes + posteriors from call logs
  02_behavioural_pilot.py        trajectories and clause effects on one open-weight model
  02b_smoke_report.py            PASS/FAIL gates on the pilot
  02c_localise_events.py         clause localisation at detected events, on the recorded model
  02d_scan_clauses.py            every clause of every step -> a development registry (deletion contrast)
  03_record_activations.py       reference vectors, all prefixes, registry events
  04_analyse.py                  selection index + candidate-token analyses
  05_feasibility.py              sample-size estimate from pilot yields
  lint_seed_families.py          mechanical checks on new MORIARTY seed families
  06_patching.py                 patching sanity checks, cross-cell patch (primary causal test),
                                 subspace patch (secondary)
data/
  episodes_exploratory.json                 40 Dataset 1-2 episodes (public repo)
  episodes_confirmatory_reconstructed.json  14 confirmatory episodes + Agent-B posteriors, rebuilt from logs
  triggers_pilot.json                       original pilot trigger records
  neutral_registry.json                     DRAFT per-event replacements for the 5 strict positions
  seeds_v5_draft.json                       30 drafted seed families (f27-f56): linted, NOT audited, NOT gated
tests/                                      37 tests, including end-to-end runs of the scripts on a random tiny model
```

## Running it

**`RUNBOOK.md` has the commands in order, with the gate after each step.**
The short version:

```
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests
python scripts/00_verify_reconstruction.py
python scripts/00b_synthetic_validation.py
python scripts/02_behavioural_pilot.py --model Qwen/Qwen2.5-7B-Instruct --format both
python scripts/02b_smoke_report.py results/pilot
python scripts/02d_scan_clauses.py --model Qwen/Qwen2.5-7B-Instruct          # development events
python scripts/03_record_activations.py --model Qwen/Qwen2.5-7B-Instruct --show-markers
python scripts/03_record_activations.py --model Qwen/Qwen2.5-7B-Instruct ...  # see RUNBOOK
python scripts/04_analyse.py
python scripts/06_patching.py --model Qwen/Qwen2.5-7B-Instruct ...
```

Model runs need a GPU with ~16 GB for a 7B model in bf16. On Colab, open
[`notebooks/moriarty_geometry_colab.ipynb`](notebooks/moriarty_geometry_colab.ipynb)
([direct link](https://colab.research.google.com/github/moloodbahar/moriarty-geometry/blob/master/notebooks/moriarty_geometry_colab.ipynb))
with a T4 GPU runtime.

The shipped registry has 5 events from 3 families (1 capture). No test can
reach p <= 0.05 with fewer than 6 held-out families, so everything computed
on it is development material and the scripts say so. `02d_scan_clauses.py`
builds a larger development sample from the existing episodes. See
DECISIONS.md §5 and §6 for what changed on 2026-11-12, what is still
undecided, and what has not been run.

## Provenance

`data/episodes_confirmatory_reconstructed.json` was produced by
`scripts/01_reconstruct_confirmatory.py` from the confirmatory run's call
logs. Its Table-2 quantities reproduce the manuscript to rounding (Dov
clause +0.865 vs 0.87). Two Appendix C clause texts in the manuscript are
paraphrases of the measured text; see DECISIONS.md §1.
