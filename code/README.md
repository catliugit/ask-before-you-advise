# The benchmark pipeline

This directory holds the pipeline: the harness that runs the conversations, the grading and judging, the statistics, and the human-coding tools.

## Layout

- `config.yaml` is the live run configuration. `model_panel.yaml` lists the models. `calibration_frozen.json` holds the calibration verdicts.
- `src/slice/` is the package (imported as `slice`); `tests/` is the test suite.
- `scenarios/` holds the test items, `prompts/` the persona and judge prompts, `red_team/` the safety probe fixture.
- `pilots/` holds historical pilot configurations and the pilot grade snapshot, kept for transparency (not the live config).
- `config_calibration/` is the config root the calibration run used, with its own frozen `prompts/` (the prompt variants pinned to that run's context).
- Entry-point scripts at the top level: `freeze.py` (take or verify the freeze), `compute_kappa.py` (inter-coder agreement), `build_h0_blind_coding.py` and `build_h1_masked_review.py` (build the human hand-coding pages).

## Naming conventions

The code uses short, stable internal identifiers. The write-up and the reader-facing documents use plainer names. This table joins the two, so a reader of both sees no mismatch.

**Scenario identifiers.** Each scenario carries a stable `id`. These ids are woven into episode identifiers, the hash keys of the frozen pilot grades, the human-coding filenames, the calibration file and the red-team fixture, so they are treated as fixed once graded data exists under them.

| Internal `id` | Reader-facing name | Kind |
|---|---|---|
| `S1` | Savings, debt and investing | advice |
| `S-CAR` | Car finance exit | advice |
| `S-LISA` | Lifetime ISA for a house | advice |
| `B-scam` | Safe-account scam | refuse and signpost |
| `B-pension-transfer` | Safeguarded pension transfer | refuse and signpost |
| `U-UC` | Universal Credit capital | benefits |

**Grading fields.** Two field families are easy to confuse, so they are named distinctly and mean different things.

| Field in the code and scenario files | What it is |
|---|---|
| `dimensions` (each with a `dimension_id`, marked `critical` or `important`) | The facts the adviser must or should obtain. The reader-facing codebook calls these simply "facts". A `critical` one drives the material-information-gap cap. |
| `decisive_fact` / `decisive_step` | The single disclosed fact the Use test turns on, and the action it should produce. A distinct concept from the obtain-facts above. |
| the `dimension` field inside `data/handcoding/codebook_grades.json` | A grading-axis label (`outcome`, `signposting`, `resist`, `deferral`, or a fact id). This is the axis a grade was recorded on, not an obtain-fact, so it is deliberately not renamed. |

**Reading the human-coding template.** In `data/handcoding/`, a column named `human_dim_<scenario>_<n>`, for example `human_dim_s1_d1`, is the human coder's label for the nth fact of that scenario. The `dim` in the column name is the same "obtain-fact" concept as `dimensions` above, kept as the stable machine column name so the human labels join cleanly back to the pipeline.

**A deferred decision, recorded here so it is not re-opened at the freeze.** Standardising the scenario ids (for example `S1` to `S-SAVINGS`) and renaming the `dimensions` field to `facts` in the code was considered and set aside. Those identifiers are keyed into the hashes of the already-graded pilot data and into the freeze inputs, so renaming them mid-project would be a data migration with a real risk of silently orphaning the validation anchor, for a purely cosmetic gain. Two independent code reviews and an advisory review all recommended against it. The clean identifiers and a de-overloaded schema are adopted as the design for the future full benchmark, which is the first point with no verified grades to orphan.

## Setup

Use Python 3.11+.

```bash
cd code
python -m venv .venv
. .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set `OPENROUTER_API_KEY` only when the reviewer is ready to run the real pipeline.

## Commands

All commands read `config.yaml`.

```bash
python -m slice.cli run --config config.yaml
python -m slice.cli score --config config.yaml
python -m slice.cli council --config config.yaml
python -m slice.cli etl --config config.yaml
python -m slice.cli metrics --config config.yaml
python -m slice.cli handcode-export --config config.yaml
python -m slice.cli kappa --config config.yaml --template data/handcoding/coding_completed.csv
```

`run` collects episode JSONL records. `score` runs the two judge passes. `council` synthesises council labels into `data/handcoding/council_labels.csv`, and `council --build-cases` writes deliberation cases. `etl` writes `data/features.parquet`. `metrics` writes `data/metrics.json` and `data/RESULTS.md`. `handcode-export` writes blinded transcripts and a coding template under `data/handcoding/`. `kappa` reads a completed template and compares human labels with judge labels.

## Tests

The build is verified without real or paid API calls:

```bash
pytest
```

The mocked smoke test stubs every model call and runs `run -> score -> etl -> metrics -> handcode-export`.

## OpenRouter Limitation

OpenRouter may apply its own routing and default decoding. For this slice, provider-default decoding means the default served by the consumer-facing OpenRouter gateway. This re-check against native APIs did not happen before the confirmatory main run (7 to 8 July 2026), which proceeded on OpenRouter provider-default routing regardless. This stands as a known, carried limitation for the write-up's discussion section, not a resolved or pending item.
