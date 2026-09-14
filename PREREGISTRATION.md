# Preregistration: Representational correlates of interpretive capture in open-weight observers

**Study title.** From behavioural capture to representational geometry: testing whether validated MORIARTY capture and release positions have a linear and causal counterpart in open-weight language models.

**Principal investigator.** Molood (Melody) Arman, independent researcher, Lyon, France.

**Version.** 2.0 (draft, 2026-09-14). Supersedes 1.0. Changes are listed in DECISIONS.md §2. Not yet frozen: the sample size, splits and outcomes are fixed after the feasibility pilot (§3.2). This document is hashed by the analysis scripts; if it changes after data collection begins, all downstream results are demoted to exploratory and reported as such.

**Status of prior work.** MORIARTY (stage 1) established, on `gpt-4o-mini`, that hidden-goal interpretation can be measured as a trajectory and that entry into and exit from a wrong-dominant trajectory localise to single clauses or two-clause spans. All thresholds and filters used below are carried over unchanged from that work unless explicitly stated.

---

## 1. Background and rationale

MORIARTY measures an observer model's distribution over four candidate hidden goals after every step of a six-step narrative generated under a known hidden goal. In the confirmatory dataset, every jointly detected wrong-dominant event agreed across two observer-information conditions in both step and wrong-goal identity (4/4), one capture trigger and four release clauses passed a strict cross-observer filter, and two further captures localised to two-clause spans.

Those results are output-level. They implicate spans of text in a distributional shift; they say nothing about what happens inside the model. Two findings in systems neuroscience motivate a specific representational prediction. Miller, Brincat and Roy (2026) argue that flexible cognition requires an active mechanism selecting which of many possible configurations is live at a given moment. Li, Chrysanthidis, Brincat, Rose and Miller (2026) report that prefrontal ensembles reorganise their representational subspaces at the point of commitment in a value-based choice: before the decision, options are coded by order of presentation; after it, by chosen versus rejected.

MORIARTY provides an unusual opportunity to ask whether an analogous reorganisation occurs in a language model, because the permutation design already dissociates the two candidate coding schemes. If goals are coded by **letter position**, rotating the labels changes the geometry. If they are coded by **semantic identity**, the geometry is rotation-invariant. If, after capture, the representation reorganises into **dominant versus rejected**, that structure should be rotation-invariant and should be weak or absent before capture.

**Scope note carried from prior work.** Capture is a descriptive trajectory shape, not a demonstrated inference error: under designed misdirection, wrong-goal dominance may be evidentially warranted. Nothing in this study tests rationality, and nothing in it bears on consciousness. The neuroscience findings above motivate the predicted shape; they are not evidence for it, and no claim of mechanistic homology between cortex and transformers is made or tested.

---

## 2. Hypotheses

Each hypothesis is stated with its predicted direction and its falsifying outcome.

**H1 — Behavioural transfer.** Validated capture and release positions reproduce on open-weight observers.
*Predicted:* at a validated position, the clause produces a probability change of ≥ 0.15 in the same direction on ≥ 3 of 4 tested model families.
*Falsified if:* fewer than 3 of 4 families show the effect. H1 is a gate (see §6): if it fails, H2–H4 are not tested and the study is reported as a failure of trigger transfer.

**H2 — Linear decodability tracks the behavioural trajectory.** A linear probe trained to decode the currently dominant goal from residual-stream activations shows a signature time-course at validated positions: the wrong-goal direction rises at the capture step, remains elevated through the committed-wrong run, and falls at the release step.
*Predicted:* mean decoded wrong-goal evidence is higher at committed-wrong steps than at pre-capture steps, and higher at pre-release than post-release steps, in both contrasts, at the preselected layer.
*Falsified if:* either contrast is null or reversed.

**H3 — Content becomes readable across presentation positions after commitment (Li et al. analogue).** A decoder trained to recover the *content* of the dominant candidate when it appeared at list position i generalises to prefixes where the same content appeared at position j ≠ i, in held-out families, more strongly after the decisive cue than before; unselected candidates' content does not show the same increase.
*Predicted:* cross-position balanced accuracy for dominant-candidate content increases from pre-event to at/after-event prefixes; the increase exceeds that for non-dominant candidates.
*Falsified if:* no increase, or an equal increase for non-dominant candidates.
*Secondary (geometry-level):* CCGP and parallelism for the dominant-vs-rejected dichotomy, matched on goal identity, increase across the event step more than for the list-position dichotomy. This uses the DECOUPLED format, in which list position and answer letter are crossed; the native rotation format cannot separate them and is not used for H3.

**H4 — Paired activation patching reconstructs and removes the clause effect.** With h_P (clause present) and h_N (neutral replacement) at the same prefix, and a subspace U learned from paired differences on training families and frozen on development families:
*H4a (restore):* h_N + UUᵀ(h_P − h_N) raises probability on the event's target goal relative to h_N, beyond a matched random subspace of the same rank.
*H4b (remove):* h_P − UUᵀ(h_P − h_N) lowers it relative to h_P, beyond the random control.
*Also tested:* the effect follows the semantic goal when its answer letter is changed (decoupled format), and does not appear for inert-clause edits.
*Falsified if:* neither patch differs from its matched random control. No universal averaged "wrong-goal direction" is assumed; the donor is always the event's own paired difference.

**Exploratory (not confirmatory).** Whether the identified direction generalises across goal families; whether release clauses act through the same direction as capture clauses with opposite sign; whether layer of peak decodability varies systematically with model depth.

---

## 3. Materials

### 3.1 Episode generation

The frozen MORIARTY pipeline is used unchanged: seed families of one world, three characters with persistent moral cores, and four covert goals in parallel form, rotated so each serves once as the true goal; Check 0 premise-prior gate at n = 100 with no goal above 0.75 of picks and no dead goal; author generation at temperature 0.8, one step per call, six steps; Check 1 goal consistency (no CONTRADICTS, ≥ 40% ADVANCES, judge calibrated at ≥ 7/8 with perfect CONTRADICTS recall); Check 2 leakage (a₁ − prior ≤ 0.20) and reachability (a₆ ≥ 0.80); Check 3 constraint inferability (≥ 0.60).

**Change from prior work:** the author agent is `gpt-4o-mini` as before, but the confirmatory family template is broadened. Prior confirmatory families shared one manipulation-style template; this study requires at least four distinct manipulation styles, specified before generation:
(i) misattributed initiative — a character is made to appear the originator of a plan;
(ii) misattributed urgency — a character's eagerness or anxiety is made to look causal;
(iii) misdirected target — attention is drawn to the wrong person being manoeuvred;
(iv) misdirected mechanism — the right target with the wrong route.
Families are assigned to styles before generation and the style is recorded as a covariate.

### 3.2 Sample size

The confirmatory sample is **not fixed in this version**. The confirmatory run yielded 4 jointly detected capture events but only 1 strict single-clause trigger from 28 generated episodes; those are different denominators, and version 1.0 conflated them. A 48-episode feasibility pilot (12 new families, 3 per manipulation style, 4 rotations each) is run first; yields are counted separately for usable episodes, joint events, strict single-clause captures, two-clause spans, committed runs and releases (`analysis/feasibility.py`), bootstrapped over families, and the sample size, splits, primary outcomes, layer-selection rule and patch settings are frozen only then. Until that freeze, all analyses on the existing 14 confirmatory episodes are development work.

No formal power analysis is performed, for the same reason as in prior work: the effect-size distribution for representational measures in this setting is unknown. 25 events is set as the minimum at which a mixed-effects model with goal-family random effects is interpretable, and this limit is stated as a constraint rather than justified as adequate. If generation halts at 200 episodes with fewer than 15 events, H2–H4 are reported as exploratory.

### 3.3 Models

Four open-weight instruction-tuned observers, fixed before data collection:

| Role | Models |
|---|---|
| Behavioural transfer (H1) | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, Mistral-7B-Instruct-v0.3 |
| Geometry and steering (H2–H4) | the single model with the highest H1 transfer rate; the runner-up serves as a replication target |

Substitution rule: if a listed model cannot be run at bf16 on available hardware, it is replaced by the next model in the same family at smaller scale, and the substitution is logged with its date and reason before any analysis is run.

### 3.4 Probe specification

Two formats (`prompts.py`). NATIVE is the MORIARTY probe unchanged (four letter rotations) and is used for behavioural transfer so that results remain comparable with the reference run. DECOUPLED lists the goals as a numbered list and gives the answer mapping separately; list order and mapping are varied independently in a 4 × 4 balanced design (16 cells), and a read-out point after the candidate list but before the mapping is recorded. DECOUPLED is a new instrument: its equivalence to NATIVE on the clause effect is measured in the pilot, and a pre-declared decision rule applies — if the present-minus-neutral effect under DECOUPLED is less than half its NATIVE value on the transferring positions, H3 and the letter-vs-goal patching test are reported as exploratory. **Change:** because the observer is open-weight, the answer distribution is computed over the full vocabulary and the A–D mass is renormalised from exact probabilities rather than from a top-20 truncation. Coverage — the fraction of total probability mass falling on the four labels before renormalisation — is recorded per call and reported; no minimum is imposed, since the quantity is now exact, but calls with coverage below 0.50 are flagged in the results as a robustness subgroup.

Derived quantities are unchanged: qₜ = pₜ(g\*); Wₜ = maxᵍ≠g\* pₜ(g); Hₜ normalised entropy; Jensen–Shannon divergence between consecutive distributions.

Event definitions are unchanged: uncertainty creation ΔH ≥ 0.15; wrong entry; wrong collapse; committed-wrong run (qₜ ≤ 0.20, Wₜ ≥ 0.70, Hₜ ≤ 0.50 on consecutive steps); resolution (true-goal gain ≥ 0.30, decreasing entropy, true-goal dominance, ≥ 3/4 rotation agreement).

### 3.5 Clause conditions

At every validated position, three conditions are run:

1. **Present** — the step as written.
2. **Neutral replacement, primary and secondary** — the trigger clause replaced by one of two length-matched (±20%), event-neutral clauses that mention the same character and introduce no goal-content words (`neutral.py` checks; judge review for grammaticality and absence of new goal-relevant information). The secondary replacement tests dependence on the control wording. Replacements are written per event and frozen by hash before measurement. This replaces the previous eight-sentence generic bank, which was world-incongruous.
3. **Inert control** — a non-trigger clause from the same step (the scene-setting clauses that prior work showed to be inert or opposing) replaced by its own length-matched neutral version.

Deletion is retained as a secondary condition for continuity with prior results.

Neutral replacements are written before any measurement, by a procedure recorded in the repository, and checked by a judge for (a) grammaticality in context, (b) no introduction of new goal-relevant information, (c) token length within ±20% of the original.

---

## 4. Procedure

**Recording.** For every episode × prefix t ∈ {0…6} × rotation ∈ {0…3} × condition, record:

- residual-stream activation at the **answer position** (the position at which the single answer letter is generated), at every layer;
- residual-stream activation at the final token of each of the four candidate-goal spans in the prompt, at every layer;
- the exact next-token distribution restricted to A–D, and coverage;
- token indices for the trigger clause span, for later alignment.

Activations are stored in bf16 at two read-out points: `after_candidates` (before the answer mapping in DECOUPLED) and `answer`. Implementation is plain `transformers` hooks (`probe.py`), with the final-layer logit-lens invariant asserted on every call.

**Splits.** Episodes are partitioned at the **goal-family** level, not the episode level, into a training split (probe fitting, layer selection, hyperparameters) and a held-out test split (all confirmatory tests). The split is drawn once, with a fixed seed recorded here, before any activation is inspected. No episode from the test split contributes to probe training, layer selection, or direction extraction.

**Layer selection.** The test layer is chosen as the layer of peak dominant-goal decoding accuracy **on the training split only**. The full layer curve is reported for all analyses; the confirmatory test uses the preselected layer.

---

## 5. Analysis plan

### 5.1 H1 — behavioural transfer

Per validated position and model, compute the change in probability on the target goal between the present and neutral-replacement conditions, permutation-averaged. A position transfers to a model if the change is ≥ 0.15 in the predicted direction. Report the transfer matrix (positions × models) in full. Family-level clustering is handled by reporting per-family as well as pooled rates.

### 5.2 H2 — decodability time-course

A multinomial logistic probe is fitted on training-split activations at the preselected layer to predict the currently dominant goal, with rotation included so that the probe cannot exploit letter position. On test-split episodes, the probe's evidence for the wrong goal is extracted at each step.

Two preregistered contrasts, each tested with a linear mixed-effects model with random intercepts for episode nested in goal family:

- committed-wrong steps versus pre-capture steps;
- pre-release versus post-release steps.

Two-sided α = 0.05, Holm correction across the two contrasts. Effect sizes with 95% confidence intervals are reported regardless of significance.

### 5.3 H3 — cross-position content decoding, then geometry

Primary (`analysis/geometry.cross_position_generalisation`): at the `after_candidates` read-out point in the DECOUPLED format, train a decoder to recover the goal identity of the dominant candidate from prefixes where it appeared at list position i; test on held-out families at position j ≠ i. Compute the same for non-dominant candidates. Contrast pre-event vs at/after-event prefixes. Test: paired difference across events, with a null from 1,000 label shuffles within event × prefix.

Secondary (Bernardi et al. 2020): CCGP and parallelism for the dominant-vs-rejected dichotomy with conditions defined by goal × position and pairs matched on goal identity; the list-position dichotomy as control. Reported with the same shuffle null. These estimates are kept secondary until the pilot shows they are stable at the available n.

### 5.4 H4 — paired patching (replaces steering)

Paired differences d_e = h_P − h_N are computed per event on the training families at every candidate layer; a subspace U (rank r ∈ {1, 2, 4, 8}) is learned by SVD of the stacked differences (`analysis/patching.learn_subspace`). Layer and rank are chosen by the mean restore effect on development families and then frozen. On held-out events: restore = h_N + UUᵀd_e, remove = h_P − UUᵀd_e, evaluated by the probability on the event's target goal after re-running the forward pass from the patched layer. Controls at the same rank: random subspaces with the injected vector rescaled to the learned injection's norm (10 per event), patches at inert-clause edits, unpatched baselines. Fluency: coverage recorded for every patched call; calls below 0.50 are reported as degenerate and excluded from flip rates (`analysis/patching.summarise`).

**Preregistered criterion:** mean Δp_target for restore exceeds the matched random control by ≥ 0.15 with a 95% CI excluding zero, on ≥ 15 held-out events; the same, with opposite sign, for remove.

---

## 6. Decision rules and stopping

1. Generate episodes until the sample criterion in §3.2 is met.
2. Run H1. **Gate:** if fewer than 3 of 4 model families transfer at ≥ 1 validated position, stop. Report the transfer failure, its implication that prior triggers may reflect author–probe self-legibility, and do not run H2–H4.
3. If the gate passes, run neutral-replacement and inert controls behaviourally, on all validated positions, before recording activations.
4. Record activations; run H2, then H3 (DECOUPLED format).
5. Run H4 last; its layer/rank selection is done on development families only.
6. Only after the feasibility pilot: freeze sample, splits and outcomes; hash this document.

No confirmatory analysis is run more than once. If code errors require re-running an analysis, the error, the fix and the date are logged, and the re-run is reported.

---

## 7. What would change the interpretation

- **H1 fails.** The most informative negative outcome available. It would indicate that validated clause triggers are properties of a model–text pairing rather than of the text, and would redirect the programme toward cross-model trigger discovery rather than representational analysis.
- **H1 passes, H2 fails.** Behaviour transfers but is not linearly decodable at the answer position. Candidate explanations to be tested exploratorily: the relevant information lives at goal-span positions rather than the answer position; the code is non-linear; the effect is distributed across layers.
- **H2 passes, H3 fails.** There is a decodable wrong-goal signal but no reorganisation into dominant-versus-rejected geometry. This would be evidence against the specific analogy to Li et al. and should be reported as such, without softening.
- **H3 passes, H4 fails.** The geometry is present but the paired patch does not move the answer beyond a random subspace. Reported as correlational, with the patching null stated plainly.
- **All pass.** The claim supported is narrow: in this model, at these validated positions, a linear direction tracks and can induce wrong-goal dominance, and the representation reorganises at commitment. It is not a claim about cortex, about consciousness, or about models in general.

---

## 8. Known limitations, stated in advance

- The author agent remains `gpt-4o-mini`, so episodes carry whatever inductive biases that model's misdirection style imposes. The observer is a different model family, which addresses the self-legibility confound for the probe but not for generation.
- The goal space remains four experimenter-specified candidates. Nothing here tests free-text or hierarchical goal inference.
- Capture remains descriptive. No normative posterior is computed in this study, so a "wrong" dominant goal may be the evidentially warranted reading.
- Patching results are sensitive to layer and rank; a positive result at one configuration is not evidence of a general mechanism, and the claim is conditional reconstruction with the event's own donor, not donor-free creation of an interpretation.
- The confirmatory episodes shipped with the repository were reconstructed from the run's call logs; two Appendix C clause texts in the manuscript are paraphrases of the measured text (DECISIONS.md §1). Analyses use the measured text.
- Parallelism and CCGP were developed for neural population recordings with different noise structure; their behaviour on transformer residual streams at these sample sizes is not well characterised, and the shuffle-based null is the primary safeguard.

---

## 9. Outputs and availability

The repository will contain: seed families and generation scripts; neutral-replacement clauses and their judge verdicts; activation-recording code with pinned library versions; the fixed split seed; probe, CCGP and steering analysis scripts; full layer curves for every analysis; all request and response logs with SHA-256 input hashes; and this preregistration, hashed by the analysis entry point.

Deviations from this document will be listed in a dated deviations section in the final manuscript, with each deviation's effect on the claim status stated.

---

## References

Bernardi, S., Benna, M. K., Rigotti, M., Munuera, J., Fusi, S., & Salzman, C. D. (2020). The geometry of abstraction in the hippocampus and prefrontal cortex. *Cell*, 183(4), 954–967.

Li, H., Chrysanthidis, N., Brincat, S. L., Rose, J., & Miller, E. K. (2026). Neural subspace reorganization reflects value-based decision-making. *iScience*.

Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Inference-time intervention: eliciting truthful answers from a language model. *Advances in Neural Information Processing Systems*, 36.

Marks, S., & Tegmark, M. (2023). The geometry of truth: emergent linear structure in large language model representations of true/false datasets. arXiv:2310.06824.

Miller, E. K., Brincat, S. L., & Roy, J. E. (2026). Analog cognition and consciousness. *The Journal of Neuroscience*, 46(33), e0711262026.

Park, K., Choe, Y. J., & Veitch, V. (2023). The linear representation hypothesis and the geometry of large language models. arXiv:2311.03658.
