#!/usr/bin/env python3
"""Rebuild data/handcoding/transcripts.jsonl from episodes.jsonl.

The coder pack's transcripts.jsonl holds raw model output and is kept out of
the replication package, so a fresh clone has the manifest and the coding
template but nothing to read. This reconstructs just that one file.

It is deliberately narrower than slice.handcode.export_handcode_pack: it takes
the pack membership, module, scenario, variant and repeat from the *committed*
coding_template.csv and only pulls `transcript` from episodes.jsonl. That way
the frozen manifest and template are never rewritten and the sample cannot
drift, whatever the sampler would do today.

Codes are recovered with the same stable_code / duplicate_code functions the
sampler used, so the mapping is exact rather than inferred.

Run from code/:
  ./rebuild_coder_transcripts.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parent
if sys.version_info < (3, 11):
    venv_python = CODE_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python), *sys.argv])

SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from slice.handcode import duplicate_code, stable_code

PACK_FIELDS = ("code", "module", "scenario", "variant", "repeat")

# episodes.jsonl carries speaker/role pairs; the coding tool renders speakers.
# test_model is the graded model, and the persona/user split is worth keeping
# because it distinguishes a scripted opener from simulated-user pushback.
_SPEAKERS = {"user", "persona", "assistant"}


def _speaker(turn: dict[str, Any]) -> str:
    speaker = str(turn.get("speaker") or "")
    if speaker == "test_model":
        return "assistant"
    if speaker in _SPEAKERS:
        return speaker
    role = str(turn.get("role") or "")
    return role if role in _SPEAKERS else "unknown"


def rebuild(
    *,
    episodes_path: Path,
    template_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    """Write transcripts.jsonl. Returns (written, missing)."""

    template_rows = list(csv.DictReader(template_path.open(encoding="utf-8", newline="")))
    if not template_rows:
        raise SystemExit(f"{template_path} has no rows")

    # One episode can back two pack codes: its own, and a duplicate drawn for
    # intra-rater reliability. Map both.
    by_code: dict[str, dict[str, Any]] = {}
    with episodes_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            episode_id = episode.get("episode_id")
            if not episode_id:
                continue
            by_code[stable_code(episode_id)] = episode
            by_code[duplicate_code(episode_id)] = episode

    written = 0
    missing: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in template_rows:
            code = row["code"]
            episode = by_code.get(code)
            if episode is None:
                missing.append(code)
                continue
            payload = {field: row.get(field, "") for field in PACK_FIELDS}
            payload["transcript"] = [
                {"speaker": _speaker(turn), "text": turn.get("text") or ""}
                for turn in (episode.get("transcript") or [])
                if isinstance(turn, dict)
            ]
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1

    if missing:
        print(f"WARNING: {len(missing)} pack codes had no episode: {', '.join(missing[:5])}")
    return written, len(missing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", default="data/episodes/episodes.jsonl")
    parser.add_argument("--template", default="data/handcoding/coding_template.csv")
    parser.add_argument("--output", default="data/handcoding/transcripts.jsonl")
    args = parser.parse_args(argv)

    written, missing = rebuild(
        episodes_path=Path(args.episodes),
        template_path=Path(args.template),
        output_path=Path(args.output),
    )
    print(f"wrote {args.output} ({written} transcripts, {missing} missing)")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
