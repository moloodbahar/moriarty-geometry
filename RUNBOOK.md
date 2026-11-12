# Runbook

Run the steps in order. Each has a gate; stop at the first gate that fails and
fix that before going on. Commands assume the repository root, with
`export PYTHONPATH=src`. Replace `$MODEL` with the observer you record from
(default `Qwen/Qwen2.5-7B-Instruct`). Run times are rough estimates for a
7B model on a T4; none has been measured.

Two facts frame everything below.

* **No test can reach p ≤ 0.05 with fewer than 6 held-out families.** The
  test is a family-level sign-flip; with n held-out families its smallest p
  is 2^-n. At the 50/20/30 split that means about 20 families with events.
* **The shipped registry has 5 events in 3 families.** It can show whether
  the pipeline runs on your model and nothing more. Step 2 builds a larger
  development sample from the 54 existing episodes. That sample is
  exploratory (DECISIONS.md §6.6); the confirmatory sample is step 7.

## Step 0 — code and analysis validation (no GPU, about 3 minutes)

```
pip install -r requirements.txt
python -m pytest -q tests
python scripts/00_verify_reconstruction.py
python scripts/00b_synthetic_validation.py
```

**Gate.** 37 tests pass; reconstruction reports 0 new-splitter failures; the
synthetic validation prints four `PASS` lines. Before any confirmatory
analysis, re-run `00b` with `--families` set to the number of event-bearing
families you actually have. If it then fails the power check, your n is too
small, and that is the answer.

## Step 1 — behavioural pilot on the recorded model (GPU, 30–90 minutes)

```
python scripts/02_behavioural_pilot.py --model $MODEL --format both --out results/pilot
python scripts/02b_smoke_report.py results/pilot
```

First look at the line `[probe] prompt rendering:`. It must say
`system+user`. Anything else means the model's chat template rejected the
system turn and the prompt format is not the one you think it is.

**Gate.** `02b` exits 0. Its gating checks: median label coverage ≥ 0.90;
no goal above 0.75 in the no-story prior; events exist on this model; native
and decoupled q-trajectories correlate at r ≥ 0.80; each inert edit moves the
target probability by < 0.05; the two neutral wordings agree within 0.10; the
explicit cue puts the target goal at ≥ 0.80; the decoupled clause effect is
at least half the native one where the native effect transfers; at least one
registry clause transfers at ≥ 0.15.

If the explicit cue fails, the probe format is not working on this model.
If no clause transfers, the model does not show these events. Either way,
change model before doing anything else.

## Step 2 — clause scan: a development sample (GPU, 45–90 minutes)

```
python scripts/02d_scan_clauses.py --model $MODEL --out results/scan
```

Tests every clause of every step in both shipped episode files and keeps
those where adding and deleting the clause agree for one goal. Writes
`results/scan/registry_scan.json`; the contrast for these events is deletion
of the clause.

**Gate.** The last line reports how many families have events. You need
about 20 for the tests in steps 5 and 6 to be able to reach significance.
With fewer, continue anyway, but read means and per-event values only.

`scripts/02c_localise_events.py` is the narrower alternative: it localises
clauses only at the capture and release steps detected in step 1.

## Step 3 — check the read-out tokens (GPU, 1 minute)

```
python scripts/03_record_activations.py --model $MODEL --show-markers
```

**Gate (by eye).** `cand_end_k` is the last token of candidate k's text, not
a newline or the next list number. `after_candidates` is the same token as
`cand_end_3`. `post_list` is the end of the `ANSWER MAPPING:` header and not
a mapping letter. `answer` is the last token of the generation prompt. If a
marker sits on the wrong token, every read-out is displaced; do not record.

## Step 4 — record activations (GPU, 1–2 hours with the scan registry)

```
EPS="data/episodes_confirmatory_reconstructed.json data/episodes_exploratory.json"
python scripts/03_record_activations.py --model $MODEL --episodes $EPS \
    --registry results/scan/registry_scan.json --layer-step 3 --out results/activations
```

For the 5 hand-written events instead, drop `--episodes` and `--registry`.
Writes reference vectors, every prefix of every episode, and each event
under present / neutral (or deletion) / inert / explicit cue / control task.
Expect several GB; keep `results/` on Drive if you are on Colab. Use
`--mode events` or `--mode trajectory` to redo one part.

**Gate.** One `ref__*.npz` per episode; 7 trajectory files per episode; 7 or
8 files per event (8 when a second neutral wording exists).

## Step 5 — geometry (CPU, minutes)

```
python scripts/04_analyse.py --activations results/activations --out results/analysis.json
```

Read the block `G1 at post_list`, in this order:

1. `status`, and the line about held-out families. Below 6, ignore every p.
2. `positive control (explicit cue)`. If it is not detected, stop: the
   statistic cannot see a goal the prompt states outright, so nothing below
   can be read as a null. If only `positive control, same-position` is
   positive, content is readable but tied to list position even then.
3. `behavioural clause effect`. If it is near zero in this format, the
   events did not survive the change from native to decoupled.
4. `trigger effect (PRIMARY)` and `trigger minus inert edit`. Both must be
   positive with p ≤ 0.05 for `G1 supported: True`.
5. `trigger minus lexical control`. If this is not positive, the trigger
   effect is no larger than under an instruction that needs no goal
   inference, and should be read as word overlap.
6. `same-position (not a test)`. It rises under a purely position-bound
   code, so it is context, not evidence.
7. `G1 at answer` against `G1 at post_list`. An effect only at `answer` is a
   code for the action (the letter), not for the content.
8. `G2a candidate tokens`, field `wrong_commitments`. The probe was fitted
   only where the dominant goal is the true one. `top_is_dominant_wrong_goal`
   near 1 means the commitment code is truth-blind; `top_is_true_goal` above
   0.25 means it tracks truth.
9. The selected layer. If it is one of the last two or three, compare with
   the logit lens before reading the result as anything but the output.

## Step 6 — patching (GPU, 30–90 minutes)

```
python scripts/06_patching.py --model $MODEL --sanity-only --out results/patching_sanity.json
python scripts/06_patching.py --model $MODEL --episodes $EPS \
    --registry results/scan/registry_scan.json --out results/patching.json
```

**Gate for the first command.** `sanity: … PASS`. The `suffix restore` line
per event shows at which block the clause effect has moved into the shared
suffix; it is not expected to be 1.0 at early blocks.

**Reading the second, phase C (the primary causal test).** Each row is a
block; the columns are the change in probability, in a recipient cell with a
different list order, of the target goal / the goal sitting at the source's
target position / after a random vector / after the inert-edit difference /
the same-cell ceiling.

* target > position goal, and target > random and inert: the clause effect
  is carried in a form that does not depend on list position. This is the
  causal version of the Li et al. result.
* position goal > target: the code is position-bound. The Li analogue fails
  at that site and block.
* `order@post_list` is before the mapping; `order@answer` after it.
  `mapping@answer` asks the same about the answer letter: target > letter
  goal means content, the reverse means the action.
* Late blocks at `answer` are close to writing the output. Read them against
  the restore curve.

Phase B (subspace patch at the answer position) is secondary;
`--skip-subspace` omits it.

## Step 7 — more families

The 16 existing families cannot give 6 held-out families, so new ones are
needed whatever step 2 finds. "More data" means more seed families: the
MORIARTY pipeline generates four episodes per family that passes the prior
gate, and there is no count flag. `--per-seed` above 1 adds stories inside a
family, which does not add families.

1. Seed families. `data/seeds_v5_draft.json` holds 30 drafted families
   (`f27`–`f56`) in the `seeds_v4_confirmatory.json` format. They are
   unaudited drafts: they pass the mechanical lint and nothing else. Read
   them, edit or delete as you see fit, and lint any you add:
   ```
   python scripts/lint_seed_families.py data/seeds_v5_draft.json \
       --existing ../Moriarty/seeds_v4_confirmatory.json ../Moriarty/seeds_v3_1.json
   ```
2. Generate and gate, in the MORIARTY repository (a few dollars of
   gpt-4o-mini; expect fewer than half the families to pass the prior gate):
   ```
   python run_pipeline.py --name geom_batch1 \
       --seeds ../moriarty-geometry/data/seeds_v5_draft.json --skip-l1 --skip-l2
   ```
   `run_pipeline.py` calls `filter_seeds.py`, which is not in the public
   MORIARTY repository as of 2026-11-12; run it from your working copy.
3. Back here, add the new episodes and their checks report to every command.
   Episodes the report marks not usable are dropped:
   ```
   EPS="$EPS ../Moriarty/results/geom_batch1/episodes.json"
   REP="../Moriarty/results/geom_batch1/checks_report.json"
   python scripts/02d_scan_clauses.py --model $MODEL --episodes $EPS --report $REP --out results/scan
   ```
   then steps 3–6 with the same `--episodes $EPS --report $REP`.
4. Count the families that have an event toward a WRONG goal. Generate
   further batches until there are about 20. `scripts/05_feasibility.py`
   turns per-family yields into a required number of families.

## Step 8 — the confirmatory run

1. Decide the open items in PREREGISTRATION.md §10; write and freeze the
   neutral replacements; hash the preregistration (`04_analyse.py` writes
   its SHA-256 into the output).
2. Generate the confirmatory families (new ones, not those used to develop
   the instrument), then run steps 1 and 3–6 once, on a fixed seed.
3. Repeat on a second model family.

## Not yet implemented

A sweep over every token between the list and the answer; the
matched-confidence wrong-vs-right probe; sparse-autoencoder decomposition;
belief-vs-report prompts; the LoRA observer; mixed-effects models for H2.
