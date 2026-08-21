# Ask Before You Advise

**A benchmark testing whether large language models perform the duties of a UK suitability fact-find before giving consumer money advice.**

Submitted as the artefact for LDSCI7237 Artificial Intelligence Dissertation Project, AE2. MSc Data Science, Northeastern University London.

---

## If you have ten minutes

Read these four files, in this order. They are the study.

1. **`pre-registration.md`** is the method of record. It was written and committed before any data existed, and it is what the freeze locks.
2. **`grading-codebook.md`** is the instrument. It grades each model answer as if a regulated adviser had produced it, with every rule tied to a named FCA provision.
3. **`freeze_record.md`** is the proof that the method predated the results, with the instrument hash and the deviations logged since.
4. **`code/data/RESULTS.md`** is the output of the pre-registered analysis, run once.

Everything else supports those four.

---

## What the study measures

When someone asks a chatbot what to do with their money, the question is usually defective. Facts are missing, the wording pushes for a particular answer, or a decisive fact is stated and inconvenient. A regulated UK adviser owes specific duties in exactly those situations. This project turns those duties into a benchmark.

- **Ask.** Elicit the missing critical facts before recommending. COBS 9, or COBS 9A on MiFID business.
- **Resist.** Hold a correct position under a question framed for a yes, and under evidence-free pushback. Consumer Duty, COBS 2.1.1R, COBS 4.2.1R.
- **Use.** Act on a decisive fact the person has disclosed.
- **Boundary.** Refuse and signpost where the law leaves no room to advise. Scams, and safeguarded pension transfers. COBS 19.1, Pension Schemes Act 2015 s.48, SI 2015/742 reg 5.

One scenario is put to a model in three framed versions, facts withheld, facts wrapped in a leading frame, and decisive fact disclosed, with planted canary facts that make elicitation machine-checkable and a quote-grounded grading cascade.

---

## The run, in numbers

| | |
|---|---|
| Conversations | 1,500 (50 scenario versions, 10 models, 3 repeats) |
| Gradings | 6,771 (4,500 cheap panel, 2,271 frontier council) |
| Council-decided episodes | 757, about half |
| Adversarial prosecutor rows | 288 |
| Quote voids | 0 |
| Scoring failures | 0.25 per cent |
| Simulated-user leak rate | 1.13 per cent, and a leaked conversation is re-run |
| API cost | $197.74 |
| Instrument frozen | 2026-07-07T21:14:06Z, hash began `f9fde5c4`, zero drift |
| Instrument hash now | `f0f10214`, after the ninth logged deviation, still zero drift |
| Logged deviations since the freeze | 9 |
| Test suite | about 785 tests |

---

## Layout

| Path | What it is |
|---|---|
| `pre-registration.md` | The frozen methodology. The method of record. |
| `grading-codebook.md` | The grading instrument, rule by rule, anchored to FCA provisions. |
| `decision-rules.md` | The casebook of borderline calls, each settled and dated. |
| `severity-rubric.md` | How consumer harm is graded, cosmetic through to critical. |
| `freeze_record.md`, `freeze_record.json` | The freeze, its hash, and the deviations log. |
| `code/` | The pipeline. Scenarios, runner, grading cascade, ETL, analysis. |
| `code/scenarios/*.json` | The scenario bank. Every item that entered the study. |
| `code/data/` | The confirmatory run: conversations, gradings, routing, metrics. |
| `build/html-viewer/v6/` | The results explorer. Recomputes every displayed number at build time. |
| `code/data/audit/` | The 275-row blind coding review, and an independent audit of 34 boundary episodes. |
| `writeup/` | Working notes behind the results and methods chapters. |

---

## Verifying the freeze, without running anything

The claim that the method predated the results does not rest on trust. `freeze_record.json` carries the instrument hash and the exact file list it covers. Recomputing that hash over the committed code reproduces it, and the freeze covers the **analysis code** as well as the protocol, so no confirmatory number can have come from a hand-built spreadsheet.

```bash
cd code
python3.12 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python freeze.py --verify
```

It returns `OK_LOGGED_DEVIATION` with an empty `drifted_files` list, which means every
file matches what the record says it should be, and every change since the freeze is
logged with a date and a reason.

---

## Re-running the analysis, without spending anything

The full run cost $197.74 in API calls and nothing here needs it repeated. The saved conversations and gradings are included, so the analysis lane runs offline from `code/data/`.

```bash
cd code
.venv/bin/python -m slice.etl --config config.yaml
.venv/bin/python -m slice.analysis --config config.yaml
```

Requires Python 3.11 or newer. Dependencies are declared in `code/pyproject.toml`.

---

## What is deliberately not here

- **The API key.** It lives only in `code/.env` locally and has never been committed. `code/.env.example` shows the shape.
- **The working history.** Session notes, planning documents and superseded drafts are not part of the artefact and are held back.

---

## An honest note on what this shows

The graders were **calibrated** against the frontier council, never validated by it. Agreement among several models shows only that they are consistent, not that they are right, and checking an automated marker against other automated markers is circular. Validity rests on blind human coding of a stratified 275-case sample, gated on weighted Cohen's kappa with bars fixed in advance at 0.75 and 0.80, read on the lower bound of the interval.

The several layers of marking are related layers of one machine rather than independent corroboration. The Outcome grade is partly determined by the Ask and Resist labels by construction, and the report says so.

Two things about the reliability gate are worth knowing before reading any number here. The gate could not ingest a human coding at all until 21 August 2026, because its metadata check rejected every episode in the run before consulting the sampling manifest, and I found that by running the gate against a synthetic fixture built to exercise the pipeline. The repair is the ninth logged deviation. The second thing is structural, which is that the gate compares the human coder against the frontier council, and the council only graded episodes the pre-registered triggers escalated to it, so only Module A has enough overlapping cases to reach its own threshold. Until that gate clears, every quantity in the results reports as estimation rather than as a confirmed finding, and none of them has been relabelled to read otherwise.
