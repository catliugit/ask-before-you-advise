# Freeze record f9fde5c41ee0

- Instrument hash: `f9fde5c41ee0d4d13b51b70c2f504c4a1e6ba72a2950eb4f62e49b0ab8462878`
- Date stamp: `2026-07-07T21:14:06Z`
- File count: 59
- External timestamp: `local-git-commit-only` 29abcba96ae746a56be1f40554c3faf7556a3aa1

## Notes

The freeze hashes config.yaml, the scenario bank, grading prompts and docs, the model panel, pre-registration, the analysis package, and the src/slice protocol package that turns prompts and saved conversations into grades and reported numbers. It excludes only cli.py, the command entry; freeze.py is included so hash lookup and validation protocol drift changes the frozen hash.

Verification accepts only the original frozen instrument_hash and, when deviations exist, the latest logged deviation hash. Earlier deviation hashes do not verify.

Before the real freeze run, confirm the grading codebook prose is consistent with the settled reading of every pilot grade.

## Files

| Path | SHA-256 | Bytes |
|---|---:|---:|
| `code/calibration_frozen.json` | `26e521f70883` | 11247 |
| `code/compute_kappa.py` | `ed3edf682d08` | 313 |
| `code/config.yaml` | `c89af8c73106` | 5178 |
| `code/model_panel.yaml` | `807bbd002566` | 11268 |
| `code/prompts/council_coder.md` | `6df2fe8b36b5` | 584 |
| `code/prompts/judge_pass1.md` | `e9778db98ccf` | 2647 |
| `code/prompts/judge_pass2.md` | `6ad720b2c952` | 5710 |
| `code/prompts/judge_pass2_boundary.md` | `f92c14a5d720` | 1378 |
| `code/prompts/judge_resist_initial.md` | `57cf99f47b9b` | 2160 |
| `code/prompts/persona.md` | `78725ccc3a6b` | 992 |
| `code/prompts/prosecutor_pass.md` | `e3a2e2e55546` | 1441 |
| `code/prompts/rubric-clarifications.md` | `103e4e7f080a` | 2684 |
| `code/red_team/probe_fixture.jsonl` | `74a1cc2f0aac` | 27371 |
| `code/scenarios/boundary_scam.json` | `b25cafdbc8b0` | 4743 |
| `code/scenarios/car_finance.json` | `d819802adac9` | 58331 |
| `code/scenarios/lifetime_isa.json` | `a286d1334b29` | 47188 |
| `code/scenarios/pension_transfer.json` | `55f1dbb6bb27` | 14918 |
| `code/scenarios/s1.json` | `9794fe9e86f2` | 18475 |
| `code/scenarios/uc_capital.json` | `e9ee2d7938cb` | 12952 |
| `code/src/slice/__init__.py` | `85a77d8c192f` | 62 |
| `code/src/slice/_jsonl.py` | `f6cff71a180a` | 2216 |
| `code/src/slice/analysis/__init__.py` | `1b8c12adc0fc` | 450 |
| `code/src/slice/analysis/aggregate.py` | `d57c5af8ac63` | 6207 |
| `code/src/slice/analysis/constants.py` | `2f9724f1ee20` | 2270 |
| `code/src/slice/analysis/ds.py` | `2a3035232ce6` | 28543 |
| `code/src/slice/analysis/fcr.py` | `612f6e311c2b` | 25884 |
| `code/src/slice/analysis/gaps.py` | `e91d7212c6c3` | 3018 |
| `code/src/slice/analysis/inference.py` | `01505a54fa91` | 9427 |
| `code/src/slice/analysis/prr_otr.py` | `e0a4422c6a31` | 7511 |
| `code/src/slice/analysis/report.py` | `ecd2ccc8a010` | 3217 |
| `code/src/slice/analysis/severity.py` | `e0845a346ec3` | 3170 |
| `code/src/slice/analysis/specgap.py` | `b2b62b4b8174` | 6814 |
| `code/src/slice/calibration.py` | `ced513215d98` | 3357 |
| `code/src/slice/calibration_gate.py` | `ed4a1a903c0c` | 6937 |
| `code/src/slice/calibration_qualify.py` | `3a9fad638c31` | 37116 |
| `code/src/slice/canary.py` | `a74a8f693ddf` | 7510 |
| `code/src/slice/client.py` | `48f46cff6c11` | 19062 |
| `code/src/slice/council.py` | `e22ca0df56e5` | 19547 |
| `code/src/slice/escalation.py` | `0990cd6a0318` | 7418 |
| `code/src/slice/etl.py` | `d84eefbb2cba` | 26456 |
| `code/src/slice/freeze.py` | `8593b6929b14` | 40003 |
| `code/src/slice/gate.py` | `9a06059d789c` | 1085 |
| `code/src/slice/handcode.py` | `a6a52f446d29` | 37729 |
| `code/src/slice/judge.py` | `b8295429ead0` | 98513 |
| `code/src/slice/kappa.py` | `4f158ef6a1e8` | 16971 |
| `code/src/slice/kappa_gate.py` | `7ea3b01ace4f` | 70406 |
| `code/src/slice/masked_review.py` | `2dc804222e63` | 17190 |
| `code/src/slice/metrics.py` | `94e264a1f4cc` | 40404 |
| `code/src/slice/nav_aid.py` | `c44ddcfab026` | 18679 |
| `code/src/slice/persona.py` | `54250976b3f7` | 1020 |
| `code/src/slice/phase_roles.py` | `6ab188e9c24a` | 3137 |
| `code/src/slice/red_team.py` | `54d89e3daca5` | 18823 |
| `code/src/slice/resolution.py` | `3ba69c150066` | 23070 |
| `code/src/slice/runner.py` | `1be6feb71142` | 37478 |
| `code/src/slice/schema.py` | `95ec5221c842` | 51948 |
| `decision-rules.md` | `94c4aa1c025d` | 25629 |
| `grading-codebook.md` | `461d3052f952` | 52567 |
| `pre-registration.md` | `79cf368a0be8` | 59026 |
| `severity-rubric.md` | `785ac69e2af7` | 1301 |

## Deviations

| Date | File | Drifted files | Change | Why | New hash |
|---|---|---|---|---|---|
| 2026-07-07T22:11:26Z | `code/config.yaml` | `code/config.yaml` | max_concurrency raised 8 to 16 for the generation phase (single-key operational throughput knob) | Ops-only speed-up, explicitly requested mid-run; calibration validated concurrency 12 with zero failures; retries/backoff and per-episode resume make provider throttling loss-free; no prompt, grading rule, routing rule or analysed quantity is affected | `650723b2a70b` |
| 2026-07-07T22:49:10Z | `code/config.yaml` | `code/config.yaml` | max_concurrency raised 16 to 32 for the judging phase (single-key operational throughput knob) | Ops-only speed-up, explicitly requested; calibration validated judging at 24 with zero failures and jobs spread across six judge models so 32 total is roughly 5-6 concurrent per provider; retries/backoff and per-row resume make throttling loss-free; no prompt, grading rule, routing rule or analysed quantity is affected | `2ecab692148a` |
| 2026-07-08T02:13:36Z | `code/src/slice/etl.py` | `code/src/slice/etl.py` | grading_role_model_versions payload now excludes scoring-failed judgement rows (one guard clause in the payload builder) | Latent over-strictness surfaced on first frozen metrics run: 13 persistent cheap-grader parse failures (0.2 per cent, within the expected ~1 per cent flake norm) carry no observed version because they produced no grade; every affected episode escalated to the council by the pre-registered trigger and carries three clean senior gradings, so no grade is missing; the version payload documents the versions that produced grades and failure records do not belong in it; the pin assertion blocked metrics until excluded. No prompt, grading rule, routing rule or analysed quantity changes; covered by a new unit test; 784 tests green | `3e6dd3601491` |
| 2026-07-08T02:22:47Z | `code/src/slice/analysis/ds.py` | `code/src/slice/analysis/ds.py` | the Module C missing-class guard now exempts rows the pre-registered council-split rule resolved to a null class WITH a human handoff; the written correctness fallback in _movement_from_pair computes those rows | First frozen analysis run surfaced a conflict between two frozen pieces: 23 of 450 C arms (5 per cent) had all judges grade fully and agree in substance but label the class with synonyms (e.g. keep_car vs continue_agreement vs keep_paying), so exact-string majority resolution abstained to null-with-human-handoff exactly as the split rule prescribes, and the analysis guard then hard-crashed on rows the machine had deliberately abstained on. The guard now exempts only handoff-flagged rows (a genuine omission still crashes); the movement fallback it unlocks was already written in the frozen code; Use is pre-specified estimation per the pre-registration; all 23 rows remain flagged for the human lane. No grades, prompts, routing or any judged value change; new unit test; 785 tests green | `2aa3ba7112d7` |
| 2026-07-08T02:25:57Z | `code/model_panel.yaml` | `code/model_panel.yaml` | leading flag corrected to false on google/gemini-3.1-pro-preview (pre-lock draft relic, model never in the locked run panel) | The confirmatory headline aggregates over leading-flagged panel entries; the stale flag from the pre-lock draft inserted a phantom never-run model row (not_established) into the headline. Registry data correction only; no grades, prompts, routing or judged values change; metrics re-run locally at zero cost | `5735eb99b101` |
| 2026-07-08 | `pre-registration.md` | `pre-registration.md` | Status line at the top corrected: it still said working draft version 0.6 not yet frozen, which misstated the frozen instrument. Replaced with the frozen-on-7-July status and a note that post-freeze changes are logged deviations. | The header contradicted the freeze record and would have misled any reader of the frozen methodology, including the examiner. Approved 8 July 2026 during the Cat-ready handover session. No methodological content changed. | `e75be6ba9179` |
| 2026-07-08 | `code/src/slice/handcode.py` | `code/src/slice/handcode.py` | export_handcode_pack now resolves the sampling pool via _resolve_handcode_split, so when no episode carries a human-sample tag it draws the blind human sample from the confirmatory pool instead of the never-populated human_dev/human_test phases. | the export sampled only never-populated phases and would produce an empty pack; fixed to sample the confirmatory episodes per the pre-registered stratification; approved 8 July 2026 | `9be1bd216cc4` |
| 2026-07-08 | `code/config.yaml` | `code/config.yaml` | One code comment neutralized: a personal-name attribution in the trio comment replaced with a dated neutral phrasing. No key, value, or behaviour changed. | Personal references are being removed from the tree ahead of the project handover, decided 8 July 2026. Comment-only change. | `f8dd3aedbcaa` |
