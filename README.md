# moriarty-geometry

Behavioural transfer, representational geometry and paired activation
patching of *interpretive capture* on open-weight observers. This is the
stage-2/3 companion to the MORIARTY framework: it takes validated capture
and release positions and asks whether they have a linear and causal
counterpart inside a model whose activations can be read.

Read `DECISIONS.md` first — it records what was verified in the original
code, what was wrong with the earlier plan, and why this design was chosen.
`PREREGISTRATION.md` (v2, draft) holds the hypotheses, thresholds and
stopping rules; it is frozen only after the feasibility pilot.

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
  prompts.py        NATIVE (4 rotations) and DECOUPLED (16 cells) probe formats
  probe.py          open-weights probe: exact label mass, marker capture, logit lens
  trajectory.py     q, W, H, JSD and the five MORIARTY events (thresholds unchanged)
  neutral.py        per-event neutral-replacement registry with validation + hash
  analysis/
    decoding.py     family-split dominance probe, layer selection, regime contrasts
    geometry.py     cross-position content decoding, CCGP, parallelism, shuffle null
    patching.py     paired-difference subspace, restore/remove, random controls
    feasibility.py  yields per denominator, family bootstrap for sample size
scripts/
  00_verify_reconstruction.py   run first
  01_reconstruct_confirmatory.py  rebuild episodes + posteriors from call logs
  02_behavioural_pilot.py       stage-1 transfer (clause effect and trajectory, separately)
  03_record_activations.py      stage-2 recording at named markers
  04_analyse.py                 stage-3 geometry with family splits
  05_feasibility.py             sample-size estimate from pilot yields
data/
  episodes_exploratory.json                 40 Dataset 1-2 episodes (public repo)
  episodes_confirmatory_reconstructed.json  14 confirmatory episodes + Agent-B posteriors, rebuilt from logs
  triggers_pilot.json                       original pilot trigger records
  neutral_registry.json                     DRAFT per-event replacements for the 5 strict positions
tests/                                      18 unit tests; run before anything else
```

## Setup

```
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests
python scripts/00_verify_reconstruction.py
```

### Google Colab

Open [`notebooks/moriarty_geometry_colab.ipynb`](notebooks/moriarty_geometry_colab.ipynb) in Colab
([direct link](https://colab.research.google.com/github/moloodbahar/moriarty-geometry/blob/master/notebooks/moriarty_geometry_colab.ipynb)),
set the runtime to **T4 GPU**, and run the cells top to bottom. Results can be
downloaded as a zip before the session ends.

Model runs need a GPU with ~16 GB for a 7B model in bf16:

```
python scripts/02_behavioural_pilot.py --model Qwen/Qwen2.5-7B-Instruct --format both
python scripts/03_record_activations.py --model Qwen/Qwen2.5-7B-Instruct --format decoupled
python scripts/04_analyse.py
```

## Order of work

1. Verify reconstruction; freeze the neutral registry after judge review.
2. Behavioural pilot on Qwen2.5-7B-Instruct. Gate on clause-effect transfer
   and on NATIVE/DECOUPLED equivalence (rule in PREREGISTRATION §3.4).
3. Record activations; run decoding and geometry with family splits.
4. Paired patching with layer/rank frozen on development families.
5. 48-episode feasibility pilot; freeze the confirmatory design; hash the
   preregistration.

## Provenance

`data/episodes_confirmatory_reconstructed.json` was produced by
`scripts/01_reconstruct_confirmatory.py` from the confirmatory run's call
logs. Its Table-2 quantities reproduce the manuscript to rounding (Dov
clause +0.865 vs 0.87). Two Appendix C clause texts in the manuscript are
paraphrases of the measured text; see DECISIONS.md §1.
