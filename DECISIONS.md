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
