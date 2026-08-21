# Methods as run

A record of what the instrument actually did, as distinct from what the
pre-registration planned. Read from `config.yaml`, `model_panel.yaml`,
`calibration_frozen.json` and `code/data/metrics.json` at instrument hash
`f9fde5c4…8462878`.

Where the run departed from the plan, or where a planned component did not
execute, that is stated here rather than left to inference.

---

## 1. Instrument and freeze

The instrument was frozen on 7 July 2026 (`freeze_day: 2026-07-07`), covering 59
files: `config.yaml`, the scenario bank, the grading prompts, the model panel,
the pre-registration, the analysis package and the `src/slice` protocol package.
Only `cli.py` is excluded; `freeze.py` is itself inside the freeze, so a change
to the hash-validation protocol changes the frozen hash.

Two deviations are logged in `freeze_record.md`, both dated 8 July 2026:

1. `etl.py` — the `grading_role_model_versions` payload now excludes
   scoring-failed judgement rows. Thirteen persistent cheap-grader parse failures
   (0.2%, within the ~1% flake norm) carried no observed version because they
   produced no grade; every affected episode escalated to the council under the
   pre-registered trigger and carries three clean senior gradings. No prompt,
   grading rule, routing rule or analysed quantity changed.
2. `handcode.py` — `export_handcode_pack` now resolves the sampling pool via
   `_resolve_handcode_split`, so that with no episode carrying a human-sample tag
   it draws from the confirmatory pool rather than the never-populated
   `human_dev` / `human_test` phases. Without this the export produced an empty
   pack.

Verification accepts only the original frozen hash or the latest logged
deviation hash.

---

## 2. Scenarios and arms

Six scenarios, each declaring its own regulatory track and severity:

| ID | Domain | Regulatory track | Severity | Primary anchors |
|---|---|---|---|---|
| S1 | general suitability | `cobs_9a` | serious | COBS 9A.2.1R, 9A.2.13R |
| S-CAR | car finance, voluntary termination | `consumer_duty` | serious | PRIN 2A.2.8R, 2A.2.9R, CONC 8.3 |
| S-LISA | Lifetime ISA | `consumer_duty` | serious | PRIN 2A.2.8R/9R, Savings (Government Contributions) Act 2017 |
| U-UC | Universal Credit capital | `consumer_duty` | serious | PRIN 2A.2.8R, UC Regs 2013 reg 18(1), reg 72(1) |
| B-scam | safe-account scam boundary | `boundary` | critical | FCA ScamSmart, 159 bank fraud line |
| B-pension-transfer | safeguarded pension transfer | `boundary` | critical | PSA 2015 s.48, SI 2015/742 reg 5, COBS 19.1.6G |

Three regulatory tracks are exercised: `cobs_9a` (suitability proper),
`consumer_duty` (PRIN 2A) and `boundary`. The severity column records the first
derivation only; the second derivation is null throughout, which is what
suppresses the severity-concentration analysis.

Four modules map onto seven arm types: withheld (Ask), fully specified
(baseline), disclosed (Use), control (false-positive guard), placebo (spurious
sensitivity), pushback/leading (Resist) and boundary (refusal). The mark set each
arm carries is fixed by arm type, not by scenario label.

The pack contains **49 distinct scenario × variant combinations** across the four
modules.

**Not defined in this version.** The schema admits a `false_premise` variant
kind. No scenario uses it, and the pre-registration records defining it or
formally excluding it as future work.

---

## 3. Models

`model_panel.yaml` holds 32 entries across four cells (western flagship, Chinese
flagship, cheap marker, cheap panel). `swap_log` is empty — no substitutions were
made during the run.

**Ten models were tested**, three repeats per module:

anthropic/claude-opus-4.8, anthropic/claude-sonnet-5, openai/gpt-5.5,
openai/gpt-5.4-mini, google/gemini-3.5-flash, z-ai/glm-5.2,
deepseek/deepseek-v4-pro, deepseek/deepseek-v4-flash, xiaomi/mimo-v2.5,
minimax/minimax-m3.

Three carry `leading: true` — claude-opus-4.8, gemini-3.5-flash and gpt-5.5.

Versions are pinned per entry (e.g. `anthropic/claude-4.8-opus-20260528`) and the
observed version is recorded per episode, so drift between the pinned and served
model is detectable after the fact.

**Provenance gap.** `leaderboard_snapshot` is `null`. The panel definition says
the cells are filled by rank on `openrouter.ai/rankings` as at the freeze day,
with the snapshot "to be saved before confirmatory run". It was not saved. The
selection rule is therefore documented but the evidence that it was applied to a
particular ranking is not recoverable. This should be stated as a limitation on
model-selection reproducibility; the pinned versions and the panel file are
unaffected.

**Possible duplicate.** `deepseek-v4-flash` and `deepseek-v4-pro` return
byte-identical choke-point figures in both graded modules (A 0.273/0.727, B
0.625/0.375). Confirm these resolved to distinct endpoints before publishing the
per-model table.

---

## 4. Conversation protocol

| Parameter | Value |
|---|---|
| Repeats | 3 per module (A, B, C, D) |
| Turn cap | 6 |
| Persona model | `qwen/qwen3.7-max`, reasoning on |
| Test-model reasoning | `default` (not forced on or off) |
| Max tokens, test model | 4096 |
| Max tokens, persona | 2048 |
| Max concurrency | 32 |
| Retries | 3 on bad output, 3 on model call |
| Cost ceiling | 25 |

The simulated user is played by a separate model under a versioned persona prompt
(`persona-week1-v3`), which is what makes multi-turn pushback and withheld-fact
arms possible.

**The test model's reasoning setting was left at provider default.** This is the
direct cause of the `reasoning_on_vs_off` comparison being uncomputable
(`left_n = 0`): no arm of the study varied it, so there is no contrast to
estimate. Report as a design limitation, not a missing result.

---

## 5. Grading pipeline

Grading is tiered by how hard and how consequential an item is, not by grader
cost.

1. **Cheap panel.** Three graders from three distinct model families
   (`deepseek-v4-pro`, `gemini-3-flash-preview`, `minimax-m3`) grade every
   conversation first, each returning the full structured labels, a confidence,
   and the exact quote each label rests on. Confidence threshold 0.95.
2. **Escalation.** Items that fail the calibrated trigger escalate.
3. **Council.** Three senior graders (`claude-opus-4.8`, `gemini-3.5-flash`,
   `gpt-5.5`) re-grade escalated items.
4. **Adversarial prosecutor pass.** Enabled (`adversarial_prosecutor_pass: true`),
   in which the strongest available case is argued against each grading line.
   The pension scenario's prosecution settled two decision rules now recorded in
   `decision-rules.md`: the signpost-versus-practical-step line, and the
   inversion of signpost roles on scam-flavoured variants.

Prompt versions are pinned and recorded per judgement (`judge-pass1-general-v2`,
`judge-pass2-general-v3`, `judge-pass2-boundary-general-v2`,
`judge-resist-initial-v1`).

Every episode in the human-coded audit sample carries **six independent
judgements**.

**Withdrawn mark.** A separate Use mark, coded per disclosed decisive fact with
its own label set, was withdrawn before the freeze. It had no valid measurement
source: graders were never asked to code it, and the automatic text-matching that
filled the stored field proved unreliable against grader outcomes in calibration.
The duty is now read from the Outcome grade and the coded outcome class.

---

## 6. Quality gates as configured

| Gate | Parameter | Value |
|---|---|---|
| Calibration audit | per scenario | 60 |
| Calibration audit | pass required per scenario | 30 |
| Calibration | trust threshold, disagreement | 10% |
| Calibration | false-safe tolerance | **0** |
| Red team | probes | 28 |
| Red team | max pass rate | 5% |
| Red team | tripwire min fire | 80% |
| Red team | tripwire max false fire | 20% |
| Safety | false-clear n floor | 10 |
| Safety | dangerous deferral max | **0** |
| Masked review | target agreement fraction | 45% |
| Navigation aid | aided fraction | 50% |
| Neutrality | bias alarm toward AI | 33.3% |

Calibration permissions are frozen per scenario in `calibration_frozen.json` for
all six scenarios.

---

## 7. Human coding: designed and not run

The blind human-coding pack was designed as follows:

| Parameter | Value |
|---|---|
| Anchor sample | 150 |
| Audit sample | 100 |
| Effective target n | 250 |
| Duplicates for intra-rater reliability | 10% (25 items) |
| **Pack total** | **275** |
| Per-module task floors | A 10, B 10, C 10, **D 20** |
| Per-model floor | 2 |
| Stratification weights | boundary 3.0, cheap-fine-on-safety 4.0, standard 1.0 |
| Sampling seed | 20260618 |
| Dev/test split ratio | 0.7 |

**Coding is incomplete, and this is the study's binding constraint.** The
`council_vs_human` gate records `n: 0`, `verdict: INSUFFICIENT_N`, with every
statistic null on all four modules. Nineteen items had been coded at the time of
writing, against a pack of 275.

The consequence is mechanical and total: with no established kappa, every
confirmatory choke point carries `status: "demoted_kappa"`, which demotes each
one from confirmatory to estimation class, which in turn makes the confirmatory
headline `not_established` by construction rather than by evidence.

Two properties of the gate bear on how much coding is actually required. The
n floor is 2, not 250 — the binding constraint is that the kappa confidence
interval's lower bound clears the threshold, which depends far more on the
observed agreement rate than on sample size. And the gate switches to a
PABAK-based rule when one label dominates above 85% prevalence, which is the
regime Module D sits in. Neither the 250 target nor the 275 pack size appears
anywhere in the pre-registration; both are code defaults.

**An independent audit was run instead**, covering 34 boundary episodes
(`code/data/audit/llm_audit_grades.csv`). It is machine coding, labelled as such,
and it is not a substitute for the human sample — it cannot enter the kappa,
because a machine coder validated against a machine coder measures nothing. Its
value is as corroboration: it reproduces the bimodal boundary split independently
and agrees with the automated judge on 14 of 14 zero-scoring episodes across 84
judgements.

**Signposting is not human-coded at all.** The coding tool hard-returns empty for
`human_signposting`, and the term appears nowhere in `compute_kappa.py` or the
pre-registration. This is consistent with the codebook, which absorbs signposting
into the boundary deferral score rather than scoring it separately in the
suitability tiers, but the write-up should say so rather than leave a visibly
empty column unexplained.

---

## 8. Cohort as run

| Module | Fully scored | Scoring failed |
|---|---:|---:|
| A | 360 | 0 |
| B | 540 | 0 |
| C | 450 | 0 |
| D | 150 | 0 |
| **Total** | **1,500** | **0** |

`missing_cells: 0 / 1500`. Bootstrap n = 10,000, permutations n = 10,000,
bootstrap seed 20260619, permutation seed 20260618. Estimation floor 40 points;
precision target maximum width 10 points; minimum discordant 5.

**No held-out set.** `cohort_sizes` records `all: 1500`, `confirmatory: 1500`,
`development: 0`. The development cohort was never populated, so nothing was
reserved for validation and every episode contributed to the confirmatory
analysis. The handcode manifest carries a `dev_test_split_ratio` of 0.7 that was
correspondingly never exercised.

---

## 9. Deviations and gaps, collected

| # | Item | Kind |
|---|---|---|
| 1 | `etl.py` scoring-failure exclusion | logged deviation, 8 Jul 2026 |
| 2 | `handcode.py` sampling-pool resolution | logged deviation, 8 Jul 2026 |
| 3 | `leaderboard_snapshot` never saved | provenance gap |
| 4 | Test-model reasoning left at default | design gap → `reasoning_on_vs_off` uncomputable |
| 5 | `development` cohort empty | design gap → no held-out set |
| 6 | Human coding incomplete (19 / 275) | **binding constraint** |
| 7 | `severity_second_derivation` null in all six scenarios | suppresses severity concentration |
| 8 | `false_premise` variant kind undefined | recorded future work |
| 9 | Confirmatory headline names one model of ten with FCR entries | open — check against prereg |
| 10 | deepseek pro/flash identical choke-point figures | **resolved — genuine coincidence** |

Items 1 and 2 are properly logged. Items 3, 4, 5 and 8 are limitations to state.
Items 6 and 7 are actionable and would change what the study can claim.

**Item 10 is resolved.** The two deepseek endpoints are genuinely distinct:
observed versions `deepseek-v4-pro-20260423` and `deepseek-v4-flash-20260423`,
150 episodes each, and across all 150 matched episode pairs **zero produced
identical test-model output**. The two models write completely different answers
and happen to land on the same aggregate pass counts (A 3/11, B 5/8). It is a
coincidence of small denominators, not aliasing. Worth one sentence in the
write-up so a reader does not assume a duplication bug.

**Item 9 remains open.** All ten models carry per-model FCR entries and four
have a full 9/9 plain-correct count, so the restriction is not explained by
testability. It could not be resolved from `metrics.json` and should be checked
against the pre-registration's definition of the confirmatory family.
