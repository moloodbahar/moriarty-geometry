# Decision record

Date: 2026-09-14. Everything below was checked against the public MORIARTY
repository at commit `f9c34fb` and against the confirmatory run's call logs
(`episodes_calls.jsonl`, `gd_steps_agentB_calls.jsonl`,
`gd_clauses_{naive,agentB}_calls.jsonl`, `seed_priors_report_calls.jsonl`).

## 1. Verified findings about the existing code

| Claim | Status | Evidence |
|---|---|---|
| `run_pipeline.py` calls a missing `filter_seeds.py` | **Confirmed** | `run_pipeline.py:175`; file absent from the tree |
| `mech_readout.py` double-normalises the final layer | **Confirmed** | HF `Qwen2Model` applies `self.norm` then appends to `hidden_states` (modeling_qwen2.py:933–937); the readout applies `final_norm` to every entry including the last |
| Clause reconstruction changes the "present" prompt | **Confirmed, overstated** | Original splitter drops the comma at `,\s+(?=and\|but\|while\|as\|so )`. Mismatch in **5 of 11** trigger entries and **2 of 5 validated** (f04_chess_club_g3, f08_wedding_g3) — not "four of five" |
| Neutral replacements are 8 generic sentences chosen by length | **Confirmed, and worse** | The bank is office-interior text (wall clock, phone buzzing, ventilation hum) — world-incongruous in most episodes, an incongruity confound on top of the manipulation |
| Some trigger records contain shortened strings | Confirmed | `probe_openweights.run_replace` matches wrong goals by 40-char prefix |

### Additional finding: the comma loss also touches the confirmatory clause results

The original splitter fails to reconstruct **139 of 324** steps in the
shipped data (43%). Among the five confirmatory strict clauses, the "full"
clause condition for **f14_coop_g1 step 6** and **f14_coop_g4 step 2** was
measured on comma-stripped text that differs from the trajectory prefix at
the same step. The trajectory probe (steps mode) used the original text;
the clause probe used the rejoined text; so for those two events the
addition/deletion baseline is not identical to the trajectory point it is
compared with. The effect sizes there (1.00/1.00 and .43/.69) are large
enough that this is unlikely to change the qualitative result, but the
clause runs should be repeated with exact spans before the numbers are
reused.

### Additional finding: Appendix C is not verbatim for two clauses

The manuscript's Appendix C ("Full Text of Confirmatory Strict Clauses")
gives, for `f15_fishing_g2` step 2, *"Her smile faded as her gaze landed on
the tangled nets."* and for `f18_rescue_g4` step 3, *"Shirin proposed
splitting into smaller groups so each volunteer could practice a specific
role."* Neither string occurs anywhere in the run logs. The measured clauses
are:

- f15_fishing_g2 s2: *"She nodded, her smile fading slightly as she glanced at the nets, a flicker of nostalgia crossing her face as she contemplated the promises made during their toughest moments."*
- f18_rescue_g4 s3: *"How about we consider splitting into smaller groups for some of the tougher routes?"* — spoken by **Emil**, not Shirin.

The other three Appendix C clauses match the logs. The Table 2 numbers were
reproduced from the logs to within rounding (e.g. Dov clause +0.865 naive vs
0.87 in the paper), so the run is the same; the appendix text is a
paraphrase. This should be corrected in the manuscript.

## 2. Direction chosen

Two candidate plans were compared: (A) a preregistration built around
rotation-based dissociation of position vs. semantics, a universal averaged
"wrong-goal direction", and 25 strict triggers estimated from joint-event
yield; (B) a repository-first plan with a decoupled prompt format, paired
donor patching through a learned subspace, and a feasibility pilot.

**Plan B is adopted**, with A's decision rules, per-hypothesis falsification
statements and limitations merged in. Reasons, each of which is a correction
to A:

1. **Native rotations confound list position with answer letter.** Goal *i*
   at letter position (i+r) mod 4 moves both together. A cannot separate
   position coding from letter coding; B's numbered list with an
   independent answer mapping can, and exposes a read-out point
   (`after_candidates`) that cannot depend on the mapping.
2. **Sample arithmetic.** The confirmatory run yielded 4 joint capture events
   but 1 strict single-clause trigger from 28 generated episodes. At the
   strict yield, 25 triggers ≈ 700 generated episodes. A used the joint
   denominator. B's 48-episode feasibility pilot with family-level bootstrap
   (`analysis/feasibility.py`) fixes the target from measured yields.
3. **No universal direction is assumed.** Episodes' wrong goals differ in
   content; averaging their differences into one direction need not produce
   anything meaningful. B patches with the event's own paired difference
   `h_P − h_N` projected through a subspace learned on training families and
   frozen on development families. The supported claim is conditional
   reconstruction of a clause effect.
4. **Closer reading of Li et al. (2026).** The finding is cross-position
   readability of the *selected option's content*, not a binary
   chosen-vs-rejected code. `analysis/geometry.cross_position_generalisation`
   implements that test; CCGP/parallelism are secondary.

Kept from A: the H1 transfer gate with an explicit stop; a falsifying
outcome for every hypothesis; the confident-correct comparison prefixes (B
also proposes these); the limitations section; the two-channel separation
of behavioural and representational outcomes.

## 3. Things this repository fixes by construction

- Clauses are character spans (`clauses.py`); every edit asserts the
  untouched text is byte-identical. Boundaries equal the original splitter's
  on every pilot entry, so results stay comparable; reconstruction is exact
  on all 324 steps in the shipped data (test: `tests/test_clauses.py`).
- Logit lens keeps the model's final norm exactly once and asserts the
  reconstructed final-layer logits equal the model's logits (`probe.py`).
- Label mass is exact over the full vocabulary; coverage is recorded, not
  assumed.
- Neutral replacements are per event, mention the character, are
  length-matched within ±20%, introduce no goal-content words, and are
  frozen by hash (`neutral.py`, `data/neutral_registry.json`). The shipped
  registry is **DRAFT**: it must pass the judge checks (grammaticality, no
  new goal-relevant information) and be set to FROZEN before any
  measurement.

## 4. Order of operations

1. `scripts/00_verify_reconstruction.py` — confirm the fix on your data.
2. Freeze the neutral registry after judge review.
3. `scripts/02_behavioural_pilot.py --format both` on Qwen2.5-7B-Instruct.
   Gate on clause-effect transfer *and* on native/decoupled equivalence.
4. `scripts/03_record_activations.py` on the passing model.
5. `scripts/04_analyse.py` — layer/rank chosen on development families,
   frozen, then held-out evaluation.
6. Generate the 48-episode feasibility set; `scripts/05_feasibility.py`;
   freeze the confirmatory sample size, splits and outcomes; update
   `PREREGISTRATION.md`, record its hash.

## 5. Changes of 2026-11-12 (branch `geometry-fixes-2026-11`)

### 5.1 Source paper

The neuroscience result the H3 analogue rests on is Li, Chrysanthidis,
Brincat, Rose & Miller (2026), *Neural subspace reorganization reflects
value-based decision-making*, iScience, **PMID 41676500** (confirmed by the
PI on 2026-11-12).

### 5.2 Defects found in the first version of the stage-2/3 code

| Defect | Consequence | Fix |
|---|---|---|
| `04_analyse.py` gave all four candidates of a prompt cell the same activation vector (`layer_acts=h[c]`) with different `dominant` labels | identical features with different labels cannot be decoded; the CCGP / parallelism output was not interpretable | per-candidate read-outs are recorded (`h_cand`, by list position); `04_analyse.py` rewritten |
| The canonical goal index was used as "content" across families, but `goals[0]` is the true goal in all 14 confirmatory episodes | a cross-family content decoder would decode "the true-goal slot" | content is defined by context-free reference vectors of the goal texts (`probe.reference_vectors`), never by index |
| `03_record_activations.py` recorded only `after_candidates` and `answer`, although PREREGISTRATION §4 specifies the final token of each candidate span | no per-candidate analysis was possible | markers `cand_end_0..3` and `after_story` added to both prompt formats |
| Only the at-event prefix under the present condition was analysed | the pre/post contrast of H3 was never computed | every condition is recorded and the contrast is present vs neutral replacement at the same prefix |
| No patched forward pass existed | H4 could not be run | `probe.run_blocks` (block-level capture and edit) and `scripts/06_patching.py` |

### 5.3 Primary G1/H3 statistic: the cross-position selection index

`analysis/selection.py`. A ridge map is fitted on training families from the
read-out activation to the family-centred reference vector of the goal that
is dominant at that prefix, one map per list position of the dominant goal.
The selection index of an event under a condition is the alignment of the
mapped read-out with the target goal's reference minus its mean alignment
with the other goals, with the map fitted at position i applied to cells
whose target sits at j ≠ i. The effect is present minus neutral replacement
at the same prefix; the inert-clause edit gives the size of a pure edit
effect. The contrast is present vs neutral and not step t−1 vs t, because
consecutive prefixes differ in length and content as well as in the clause.

The unit is the event. Cells of the 16-cell design are repeated measures,
and read-outs taken before the answer mapping are identical across the four
mappings of a list order, so they are de-duplicated. The test is one-sided,
for both the effect and its excess over the inert edit. (The bootstrap
interval first used as the test here was miscalibrated; see §6.1.)

### 5.4 An artefact found by the null simulation, and its fix

`analysis/simulate.py` generates recordings in which a position-invariant
selected code either exists (planted) or does not (null: the clause still
changes behaviour and a position-bound content code). In its first form the
cross-position index was *detected in 40% of null simulations*. Cause: a map
fitted at position i reads whatever goal sits at i; in the cross-position
case that is a non-target goal, whose evidence the clause lowers, so the
target-minus-others difference rises without any selected code. Fix: the goal
sitting at the map's fitting position is excluded from the comparison set.
After the fix: 0 of 60 null simulations detected, planted effect detected in
all simulations at 12 families (`scripts/00b_synthetic_validation.py`).
The same-position index stays positive under the null, which is why it is
reported but is not the test.

### 5.5 Patching

Positions are addressed from the end of the sequence: a clause-present and a
clause-replaced prompt share everything after the story but not a prefix
length. Activations are captured and edited at block outputs with forward
hooks, because `hidden_states[-1]` is post-final-norm and cannot be patched.

Correction to the test plan of 2026-11-12, which said a full residual patch
at all post-clause positions "must restore 100%". That is wrong for
intermediate blocks: later blocks still attend back to the neutral clause
tokens. The sanity checks that must hold exactly are the identity patch and
the last-block answer-position patch; the full-suffix patch is reported as a
restore curve over depth.

### 5.6 Open decisions (not made in this change)

1. **Which events define the geometry sample.** PREREGISTRATION §6 gates
   H2–H4 on transfer of the gpt-4o-mini triggers (H1). The alternative is to
   detect and localise events on the recorded model
   (`scripts/02c_localise_events.py`) and report H1 transfer separately.
2. **H3 wording.** §5.3 of the preregistration describes a decoder of goal
   identity generalising to held-out families. Goal identity is not defined
   across families (5.2). The selection index is proposed as the primary
   analysis; the preregistration text must be changed before the freeze.
3. **H2.** The preregistration specifies a multinomial dominant-goal probe;
   `analysis/decoding.py` implements a shared binary probe. Unresolved.

### 5.7 Not implemented

Matched-confidence wrong-vs-right probe (G2c); sparse-autoencoder
decomposition; belief-vs-report prompts; the LoRA observer; mixed-effects
models for H2.

### 5.8 What has and has not been run

The unit tests pass (35 after §6). All scripts were executed end to end on a randomly
initialised 3-layer model (`tests/test_scripts_tinymodel.py`), which checks
code paths and file shapes only. **Nothing in this change has been run on a
real model.**

## 6. Second change of 2026-11-12: review of §5 before any model run

The PI asked for an independent second look before starting. Nothing had
been run on a model, so everything below was free to change.

### 6.1 The test in §5.3 was miscalibrated

Under an exact null (two independent draws of the same condition) the
family-clustered bootstrap rejected about 10% of the time at a nominal 5%,
with 4 and with 7 held-out families. The "0 of 60 null simulations" of §5.4
was true but flattering: that null has a negative mean.

Replaced by a **family-level sign-flip permutation test**, one-sided, events
averaged within family (`selection.family_signflip`). Measured rejection
under the exact null: 3.7% over 80 simulations at 7 held-out families. The
bootstrap interval is still printed, as description only.

Consequence for design: with n held-out families the smallest attainable p
is 2^-n. Four families cannot reach 0.05. **Six held-out families is the
working minimum**, which at the 50/20/30 split means about 20 families with
events. Scripts print this and never report "supported" below it.

### 6.2 Read-out token

`after_candidates` is the last token of candidate 4, so it mostly carries
how well candidate 4 fits the story. The primary read-out is now
`post_list`: the header token after the whole list and before the mapping
content. It has seen all four candidates and is identical across mappings.
`after_candidates` and `answer` are still recorded and analysed.

### 6.3 Two controls on the real model

* **Positive control (`cue`).** The neutral version of the step followed by
  an explicit statement of the target goal. The simulation only shows that
  the code recovers the generative model it was written for; it cannot show
  the statistic is sensitive on a transformer. If the cue is not detected by
  the cross-position index, a null on real events is uninterpretable. If the
  cue is detected by the same-position index only, content is readable but
  position-bound even when the goal is unmistakable, which is itself an
  answer to the Li question.
* **Lexical control (`ctl_*`).** The same prompts under an instruction that
  needs no goal inference ("which sentence is longest"). A trigger clause
  usually shares words with its target goal; the clause effect under this
  instruction is that overlap. The trigger effect must exceed it.

### 6.4 The primary causal test is a cross-cell patch

`scripts/06_patching.py`, phase C. The present-minus-neutral difference at
one read-out token of a source cell is added to the neutral run of a
recipient cell of the same event that has a different list order. The answer
then moves either to the target goal (position-invariant content code) or to
the goal sitting at the source's target position (position-bound code). The
same across answer mappings, at the answer position, separates content from
letter. This needs no reference vectors, no map and no training families,
and it is the causal form of the Li et al. claim. The subspace patch of §5.5
is kept as a secondary analysis (phase B).

### 6.5 Smaller changes

* Prompt rendering is explicit. `probe.render` tries system+user, then a
  user-only chat turn, then plain text, prints which one it used, and the
  patching output records it. Before, a template that rejected a system role
  fell back to plain text silently.
* Events are recorded under a second, non-cyclic Latin square of list
  orders; trajectory prefixes (on which the maps are fitted) keep the cyclic
  one. In the cyclic square a goal always has the same neighbour before it.
* Candidate-token activations are stored once per list order.
* Both shipped episode schemas load through `episodes.load_episodes`.

### 6.6 Clause scan: a development sample, not the preregistered one

`scripts/02d_scan_clauses.py` tests every clause of every step on the
recorded model and keeps those where addition and deletion agree for one
goal (gain >= 0.15, drop >= 0.10). The contrast for these events is deletion
of the clause; the inert control is deletion of the least effective other
clause of the same step. Reasons: 5 events in 3 families cannot exercise any
test (6.1), and the geometry question needs a clause that moves a goal, not
the full capture/release trajectory shape. Whether the moved goal is the
true or a wrong one is kept as `trigger_type`, which is what the
wrong-vs-right comparison needs.

Limits, stated so they are not forgotten: these events are selected on
effect size (selection in NATIVE, analysis in DECOUPLED, so the analysed
measurement is not the selected one); deletion changes discourse coherence
in a way a neutral replacement does not; and this is not the sample defined
in PREREGISTRATION §3. Results on it are exploratory. It exists to find out
whether the instrument works before families are generated at scale.

### 6.7 Still not done

Verified only on a random tiny model. Not implemented: a sweep over every
token between the list and the answer; the matched-confidence wrong-vs-right
probe; sparse-autoencoder decomposition; belief-vs-report prompts; the LoRA
observer; mixed-effects models for H2. The open decisions of §5.6 stand.
