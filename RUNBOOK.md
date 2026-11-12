# Runbook

Run the steps in order. Each has a gate; stop at the first gate that fails and
fix that before going on. Commands assume the repository root, with
`export PYTHONPATH=src`. Replace `$MODEL` with the observer you record from
(default `Qwen/Qwen2.5-7B-Instruct`). Run times are rough estimates for a
7B model on a T4; none has been measured.

What is and is not covered by the current data: the shipped registry has
5 events from 3 families, one of them a capture. Steps 1–6 run on it and tell
you whether the instrument works on your model. They cannot support a
claim. The scripts print `DEVELOPMENT ONLY` until the registry is frozen and
there are at least 15 held-out events.

## Step 0 — code and analysis validation (no GPU, about 2 minutes)

```
pip install -r requirements.txt
python -m pytest -q tests
python scripts/00_verify_reconstruction.py
python scripts/00b_synthetic_validation.py --families 12 --sims 40
```

**Gate.** 31 tests pass; reconstruction reports 0 new-splitter failures; the
synthetic validation prints three `PASS` lines. Re-run `00b` with
`--families` set to the number of event-bearing families you actually have
before any confirmatory analysis: the power line is only meaningful at your n.

## Step 1 — behavioural pilot on the recorded model (GPU, 30–90 minutes)

```
python scripts/02_behavioural_pilot.py --model $MODEL --format both --out results/pilot
python scripts/02b_smoke_report.py results/pilot
```

**Gate.** `02b` exits 0. Its gating checks: median label coverage ≥ 0.90;
no goal above 0.75 in the no-story prior; events exist on this model; native
and decoupled q-trajectories correlate at r ≥ 0.80; each inert edit moves the
target probability by < 0.05; the two neutral wordings agree within 0.10; the
decoupled clause effect is at least half the native one where the native
effect transfers; at least one registry clause transfers at ≥ 0.15.

If no clause transfers, the model does not show these events. Change model
before doing anything else.

## Step 1b — optional: localise events on this model

Only if you decide that events are defined on the recorded model
(DECISIONS.md §5.6, item 1).

```
python scripts/02c_localise_events.py --model $MODEL --pilot results/pilot
```

Output: `results/pilot/localised_events.json`, candidate registry records
with empty neutral fields. Write `neutral_primary`, `neutral_secondary` and
`inert_neutral` for the strict ones by hand, have them judged, add them to
`data/neutral_registry.json`, and freeze it before step 3.

## Step 2 — check the read-out tokens (GPU, 1 minute)

```
python scripts/03_record_activations.py --model $MODEL --show-markers
```

**Gate (by eye).** `cand_end_k` is the last token of candidate k's text, not
a newline or the next list number. `after_candidates` is the same token as
`cand_end_3`. `answer` is the last token of the generation prompt. If a
marker sits on the wrong token, every read-out is displaced; do not record.

## Step 3 — record activations (GPU, 30–45 minutes on a T4)

```
python scripts/03_record_activations.py --model $MODEL --layer-step 2 --out results/activations
```

Writes reference vectors, every prefix of every episode, and each registry
event under present / two neutral / inert conditions. About 1.3 GB at
`--layer-step 2`, 2.5 GB at 1. Use `--mode events` or `--mode trajectory` to
redo one part.

**Gate.** One `ref__*.npz` per episode; 7 trajectory files per episode;
5 files per registry event.

## Step 4 — geometry (CPU, a few minutes)

```
python scripts/04_analyse.py --activations results/activations --out results/analysis.json
```

Read, in this order:

1. `status`. `DEVELOPMENT ONLY` means the numbers below are for debugging.
2. `G1 at after_candidates`, line `cross:present_minus_neutral_primary`. This
   is the primary statistic. It must be positive with a 90% interval above
   zero, and so must `cross:trigger_minus_inert_edit`.
3. `cross:present_minus_neutral_secondary` should agree with the primary. If
   it does not, the result depends on the wording of the neutral sentence.
4. `same:` and `direct:` lines are reported for comparison. The same-position
   index rises under a purely position-bound code, so it is not evidence.
5. `G1 at answer` against `G1 at after_candidates`. An effect only at
   `answer` is a code for the action (the letter), not for the content.
6. `G2a candidate tokens`, field `wrong_commitments`. The probe was fitted
   only on prefixes where the dominant goal is the true one.
   `top_is_dominant_wrong_goal` near 1 means the commitment code is
   truth-blind; `top_is_true_goal` above chance means it tracks truth.
7. The selected layer. If it is one of the last two or three, compare against
   the logit lens before reading the result as anything but the output.

**Gate.** `supported: True` at `after_candidates` on development families
with the real sample. On the shipped 5 events this cannot be met; use the
run to check that the numbers are finite and the layer curve is smooth.

## Step 5 — patching (GPU, 20–40 minutes)

```
python scripts/06_patching.py --model $MODEL --sanity-only --out results/patching_sanity.json
python scripts/06_patching.py --model $MODEL --out results/patching.json
```

**Gate for the first command.** `sanity: … PASS`: the identity patch and the
last-block answer-position patch both reproduce the expected answer within
5e-3. The printed `suffix restore` line per event shows at which block the
clause effect has moved into the shared suffix; it is not expected to be 1.0
at early blocks.

**Reading the second.** `restore_minus_random` ≥ 0.15 with an interval above
zero is the preregistered criterion. `restore_minus_other_donor` near zero
means another event's difference works as well, so the subspace is not
content-specific. `restore_minus_inert_subspace` near zero means a generic
edit direction works as well. `norm_ratio` far from 1 means the patched
state is outside the natural range. With fewer than 3 families showing a
behavioural effect the script stops after the sanity phase and says so.

## Step 6 — sample size, then the confirmatory run

1. Generate the 48-episode feasibility set with the MORIARTY pipeline (other
   repository), run steps 1–1b on it, and count yields per family.
2. `python scripts/05_feasibility.py yields.json --target 60`, where
   `yields.json` is a list of per-family counts with the fields of
   `analysis/feasibility.FamilyYield`.
3. Decide the open items in PREREGISTRATION.md §10; freeze the registry; hash
   the preregistration (`04_analyse.py` writes its SHA-256 into the output).
4. Generate the confirmatory families, repeat steps 1–5 once, on a fixed seed.

## Not yet implemented

Matched-confidence wrong-vs-right probe, sparse-autoencoder decomposition,
belief-vs-report prompts, the LoRA observer, mixed-effects models for H2.
