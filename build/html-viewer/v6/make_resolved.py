#!/usr/bin/env python3
"""Extract the RESOLVED per-episode grade values from the study's
features.parquet into a small stdlib-readable resolved.json that build_viewer.py
consumes.

Why this exists: features.parquet carries the pre-registered *resolved* grade for
every episode (outcome_grade after gap-capping, safety-breaks, council
deliberation, the prosecutor tripwire, and human-handoff abstention). Re-deriving
those headline grades from a raw judge majority disagrees with the resolved table
on ~15% of suitability episodes, so the viewer must prefer the resolved values.

parquet needs pyarrow, which is not in the system python but IS in the study
venv, so this extractor runs under that venv and writes a plain JSON. The viewer
build itself stays stdlib-only and reads this JSON.

Run after every study run, before building the viewer:

    code/.venv/bin/python3 build/html-viewer/v6/make_resolved.py

It reads (never writes) code/data/features.parquet and writes resolved.json next
to build_viewer.py. Nothing under code/ is touched.
"""

import datetime
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent                     # .../Dissertation
PARQUET = ROOT / "code" / "data" / "features.parquet"
OUT = HERE / "resolved.json"
REGEN_COMMAND = "code/.venv/bin/python3 build/html-viewer/v6/make_resolved.py"

# The per-episode columns the viewer needs for headline grades. Everything else
# (resist behaviour, ask dimensions, signposting, routing, cost) is still read
# from the raw run files by the build; only the resolved outcome lives here.
COLS = [
    "episode_id",
    "module",
    "outcome_grade",          # resolved suitability grade (None => machine abstained)
    "outcome_class",
    "outcome_gap_capped",     # True => a required fact was left unobtained, grade capped
    "deferral_score",         # resolved Boundary score (0-3)
    "final_grade_basis",      # unanimous | cheap_consensus | deliberated-majority
                              #           | safety_break | human_handoff | prosecutor_tripwire
    "final_grade_human_handoff",  # True => referred to the study's human coding
    "final_grade_source_tier",
    # Resolved Resist-module values (same story as the grades: the build must not
    # re-derive these from a raw judge majority).
    "resist_initial",             # resisted | accepted_unsafe_course (first answer)
    "resist_pushback",            # held_firm | caved | not_triggered | not_applicable
    "pre_pushback_grade",         # the grade of the first answer, before the pushback turn
    "mechanical_disclosed_fact_use",   # USED | CONTRADICTED | IGNORED (Module C)
]


def main():
    if not PARQUET.exists():
        raise SystemExit(f"missing input: {PARQUET}")
    raw = PARQUET.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    source_mtime = datetime.datetime.fromtimestamp(
        PARQUET.stat().st_mtime, datetime.timezone.utc).isoformat()

    t = pq.read_table(str(PARQUET), columns=COLS).to_pydict()
    n = len(t["episode_id"])
    # Every field below is read VERBATIM from the parquet column of the same
    # name (with only None/bool normalisation). There is no grade logic here:
    # the pre-registered resolution (gap-capping, safety-breaks, deliberation,
    # prosecutor tripwire, human handoff) already ran in the study machine and
    # is baked into these columns. Do not add scoring logic to this file.
    resolved = {}
    for i in range(n):
        eid = t["episode_id"][i]
        resolved[eid] = {
            "module": t["module"][i],
            "grade": t["outcome_grade"][i],
            "oclass": t["outcome_class"][i],
            "gap_capped": bool(t["outcome_gap_capped"][i]) if t["outcome_gap_capped"][i] is not None else False,
            "deferral": t["deferral_score"][i],
            "basis": t["final_grade_basis"][i],
            "handoff": bool(t["final_grade_human_handoff"][i]) if t["final_grade_human_handoff"][i] is not None else False,
            "src_tier": t["final_grade_source_tier"][i],
            "resist_initial": t["resist_initial"][i],
            "resist_pushback": t["resist_pushback"][i],
            "pre_pushback_grade": t["pre_pushback_grade"][i],
            "fact_use": t["mechanical_disclosed_fact_use"][i],
        }
    payload = {
        "_meta": {
            "source": "code/data/features.parquet",
            "source_sha256": source_sha256,
            "source_mtime": source_mtime,
            "n_episodes": n,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "regen_command": REGEN_COMMAND,
            "note": "Resolved per-episode grade values, read verbatim from the parquet. "
                    "Regenerate after each study run; build_viewer.py fails if this is stale. "
                    "Do not hand-edit.",
        },
        "resolved": resolved,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    abstain = sum(1 for r in resolved.values()
                  if r["module"] != "D" and r["grade"] is None and r["handoff"])
    print(f"resolved.json written: {n} episodes, {abstain} non-Boundary machine-abstentions")
    print(f"  source sha256 {source_sha256[:16]}… mtime {source_mtime}")


if __name__ == "__main__":
    main()
