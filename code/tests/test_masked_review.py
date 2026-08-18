from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from slice.handcode import duplicate_code, stable_code
from slice.masked_review import (
    MASKED_CODER_PACK_FILES,
    export_masked_review_pack,
    flip_direction,
    masked_code,
    _h1_lock_hash,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ID = "S-mask"
ASK_ID = "income.need"
ASK_COL = "human_ask_income.need"
DIM_COL = "dim_income_need"


def test_disagreement_classification_per_scored_field_all_module_shapes(tmp_path):
    specs = [
        _spec("ep-a-profile", "A", "A-profile", h0_ask="elicited", ai_ask="unconditioned", h0_outcome="correct", ai_outcome="correct"),
        _spec("ep-a-full", "A", "A-full", h0_outcome="partial", ai_outcome="partial"),
        _spec("ep-b", "B", "B-plain", h0_outcome="correct", ai_outcome="harmful"),
        _spec("ep-c", "C", "C-leading", h0_outcome="correct", ai_outcome="incorrect"),
        _spec("ep-d-disagree", "D", "D-boundary", h0_deferral="1", ai_deferral=3.0),
        _spec("ep-d-agree", "D", "D-boundary", h0_deferral="2", ai_deferral=2.0),
    ]
    output = export_masked_review_pack(_write_fixture(tmp_path, specs, target=0.9))
    manifest = _manifest(output)
    reveal = _reveal(output)

    assert manifest["queue_composition"]["n_disagreements"] == 4
    assert manifest["queue_composition"]["n_agreements"] == 2

    profile = _reveal_for_episode(manifest, reveal, "ep-a-profile")
    assert profile["h0_label"][ASK_ID] == "elicited"
    assert profile["ai_final_grade"][ASK_ID] == "unconditioned"

    full = _case_for_episode(output, manifest, "ep-a-full")
    assert full["scored_fields"] == ["outcome"]
    assert _reveal_for_episode(manifest, reveal, "ep-a-full")["h0_label"]["outcome"] == "partial"

    assert _reveal_for_episode(manifest, reveal, "ep-b")["ai_final_grade"]["outcome"] == "harmful"
    assert _reveal_for_episode(manifest, reveal, "ep-c")["ai_final_grade"]["outcome"] == "incorrect"
    assert _reveal_for_episode(manifest, reveal, "ep-d-disagree")["h0_label"]["deferral"] == "1"
    assert _reveal_for_episode(manifest, reveal, "ep-d-disagree")["ai_final_grade"]["deferral"] == "3"
    coerced = _reveal_for_episode(manifest, reveal, "ep-d-agree")
    assert coerced["h0_label"]["deferral"] == "2"
    assert coerced["ai_final_grade"]["deferral"] == "2"


def test_null_handling_excludes_fields_and_counts_skips(tmp_path):
    specs = [
        _spec("ep-disagree", "B", "B-plain", h0_outcome="correct", ai_outcome="harmful"),
        _spec("ep-partial", "A", "A-profile", h0_ask="", ai_ask="unconditioned", h0_outcome="correct", ai_outcome="correct"),
        _spec("ep-missing-ai", "B", "B-plain", h0_outcome="correct", ai_outcome=None),
        _spec("ep-missing-h0", "B", "B-plain", h0_outcome="", ai_outcome="correct"),
    ]
    output = export_masked_review_pack(_write_fixture(tmp_path, specs, target=0.9))
    composition = _manifest(output)["queue_composition"]
    reveal = _reveal(output)

    assert composition["n_eligible"] == 2
    assert composition["skipped_missing_ai"] == 1
    assert composition["skipped_missing_h0"] == 1
    assert composition["skipped_no_comparable_field"] == 2

    partial = _reveal_for_episode(_manifest(output), reveal, "ep-partial")
    assert partial["h0_label"][ASK_ID] is None
    assert partial["ai_final_grade"][ASK_ID] == "unconditioned"
    assert partial["h0_label"]["outcome"] == partial["ai_final_grade"]["outcome"] == "correct"


def test_queue_composition_fraction_and_shortfall(tmp_path):
    exact_specs = [
        *[_spec(f"ep-diff-{i}", "B", "B-plain", h0_outcome="correct", ai_outcome="harmful") for i in range(3)],
        *[_spec(f"ep-agree-{i}", "B", "B-plain", h0_outcome="partial", ai_outcome="partial") for i in range(4)],
    ]
    exact = export_masked_review_pack(_write_fixture(tmp_path / "exact", exact_specs, target=0.4))
    exact_comp = _manifest(exact)["queue_composition"]

    assert exact_comp["n_disagreements"] == 3
    assert exact_comp["n_agreements"] == 2
    assert exact_comp["queue_n"] == 5
    assert exact_comp["achieved_agreement_fraction"] == 0.4
    assert exact_comp["n_disagreements"] + exact_comp["n_agreements"] == exact_comp["queue_n"]

    shortfall_specs = [
        *[_spec(f"ep-short-diff-{i}", "B", "B-plain", h0_outcome="correct", ai_outcome="harmful") for i in range(4)],
        _spec("ep-short-agree", "B", "B-plain", h0_outcome="partial", ai_outcome="partial"),
    ]
    shortfall = export_masked_review_pack(_write_fixture(tmp_path / "shortfall", shortfall_specs, target=0.5))
    shortfall_comp = _manifest(shortfall)["queue_composition"]

    assert shortfall_comp["n_agreements_target"] == 4
    assert shortfall_comp["n_agreements"] == 1
    assert shortfall_comp["agreement_shortfall"] == 3
    assert shortfall_comp["queue_n"] == 5
    assert shortfall_comp["achieved_agreement_fraction"] == 0.2


def test_remasking_is_unique_disjoint_and_excludes_duplicates(tmp_path):
    specs = [
        _spec("ep-source", "B", "B-plain", h0_outcome="correct", ai_outcome="harmful"),
        _spec("ep-other", "C", "C-leading", h0_outcome="partial", ai_outcome="incorrect"),
    ]
    output = export_masked_review_pack(
        _write_fixture(tmp_path, specs, target=0.9, duplicate_episode_ids=["ep-source"])
    )
    manifest = _manifest(output)
    masked_map = manifest["masked_map"]
    duplicate = duplicate_code("ep-source")

    assert masked_map
    assert all(code.startswith("M") for code in masked_map)
    assert len(set(masked_map)) == len(masked_map)
    assert not (set(masked_map) & {stable_code("ep-source"), stable_code("ep-other"), duplicate})
    assert duplicate not in set(masked_map.values())
    assert set(masked_map.values()) == {stable_code("ep-source"), stable_code("ep-other")}


def test_hold_back_blinding_and_post_lock_reveal(tmp_path):
    specs = [_spec("ep-blind", "B", "B-plain", h0_outcome="correct", ai_outcome="harmful")]
    output = export_masked_review_pack(_write_fixture(tmp_path, specs, target=0.9))
    manifest = _manifest(output)
    source_code = stable_code("ep-blind")
    masked = _masked_for_episode(manifest, "ep-blind")

    for name in MASKED_CODER_PACK_FILES:
        text = (output / name).read_text()
        assert "ep-blind" not in text
        assert source_code not in text
        assert "ai_final_grade" not in text
        assert "h0_label" not in text
        assert "post_lock_reveal" not in text
        assert "test/model" not in text

    instructions = (output / "instructions.md").read_text().lower()
    assert not re.search(r"\bai\b", instructions)
    assert "machine" not in instructions
    assert "disagreed" not in instructions

    reveal = _reveal(output)
    assert reveal[masked]["ai_final_grade"]["outcome"] == "harmful"
    assert reveal[masked]["h0_label"]["outcome"] == "correct"


def test_h1_lock_hash_is_hex_deterministic_and_coder_file_scoped(tmp_path):
    specs = [_spec("ep-lock", "B", "B-plain", h0_outcome="correct", ai_outcome="harmful")]
    output = export_masked_review_pack(_write_fixture(tmp_path, specs, target=0.9))
    manifest = _manifest(output)
    lock_hash = manifest["h1_lock_hash"]

    assert manifest["pack"] == "h1_masked"
    assert re.fullmatch(r"[0-9a-f]{64}", lock_hash)
    assert _h1_lock_hash(output) == lock_hash

    (output / "stray.txt").write_text("ignored\n")
    assert _h1_lock_hash(output) == lock_hash
    (output / "post_lock_reveal.json").write_text('{"changed": true}\n')
    assert _h1_lock_hash(output) == lock_hash


def test_flip_direction_branches():
    assert flip_direction("correct", "correct", "harmful") == "unchanged"
    assert flip_direction("correct", "harmful", "harmful") == "toward_ai"
    assert flip_direction("correct", "partial", "correct") == "away_from_ai"
    assert flip_direction("correct", "partial", "harmful") == "third_option"
    assert flip_direction("correct", "partial", None) == "no_ai_label"
    assert flip_direction("branch-covered", "branch_covered", "elicited") == "unchanged"


def test_determinism_two_runs_identical_map_hash_and_pack_bytes(tmp_path):
    specs = [
        _spec("ep-det-1", "B", "B-plain", h0_outcome="correct", ai_outcome="harmful"),
        _spec("ep-det-2", "C", "C-leading", h0_outcome="partial", ai_outcome="partial"),
        _spec("ep-det-3", "D", "D-boundary", h0_deferral="1", ai_deferral=3.0),
    ]
    config = _write_fixture(tmp_path, specs, target=0.9)
    first_output = export_masked_review_pack(config)
    first_manifest = _manifest(first_output)
    first_bytes = _pack_bytes(first_output)

    second_output = export_masked_review_pack(config)
    second_manifest = _manifest(second_output)
    second_bytes = _pack_bytes(second_output)

    assert second_manifest["masked_map"] == first_manifest["masked_map"]
    assert second_manifest["h1_lock_hash"] == first_manifest["h1_lock_hash"]
    assert second_bytes == first_bytes


def test_degenerate_no_h0_zero_disagreements_and_no_features(tmp_path):
    specs = [
        _spec("ep-degen-1", "B", "B-plain", h0_outcome="correct", ai_outcome="correct"),
        _spec("ep-degen-2", "C", "C-leading", h0_outcome="partial", ai_outcome="partial"),
    ]

    no_h0 = export_masked_review_pack(_write_fixture(tmp_path / "no-h0", specs, include_h0=False))
    no_h0_comp = _manifest(no_h0)["queue_composition"]
    assert no_h0_comp["n_eligible"] == 0
    assert no_h0_comp["queue_n"] == 0
    assert no_h0_comp["skipped_missing_h0"] == 2
    assert no_h0_comp["skipped_no_comparable_field"] == 2

    zero_diff = export_masked_review_pack(_write_fixture(tmp_path / "zero", specs))
    zero_comp = _manifest(zero_diff)["queue_composition"]
    assert zero_comp["n_eligible"] == 2
    assert zero_comp["n_disagreements"] == 0
    assert zero_comp["n_agreements"] == 0
    assert zero_comp["queue_n"] == 0
    assert zero_comp["skipped_no_comparable_field"] == 0

    no_features = export_masked_review_pack(_write_fixture(tmp_path / "no-features", specs, include_features=False))
    no_features_comp = _manifest(no_features)["queue_composition"]
    assert no_features_comp["n_eligible"] == 0
    assert no_features_comp["queue_n"] == 0
    assert no_features_comp["skipped_missing_ai"] == 2
    assert no_features_comp["skipped_no_comparable_field"] == 2


def test_masked_code_collision_raises_before_silent_dedup(monkeypatch, tmp_path):
    # Force two distinct episodes onto the same masked_code. The guard must RAISE, not silently drop
    # one item when _review_queue keys its dicts by masked_code (the cross-read's edge case).
    import slice.masked_review as mr

    specs = [
        _spec("ep-coll-1", "B", "B-plain", h0_outcome="correct", ai_outcome="harmful"),
        _spec("ep-coll-2", "C", "C-leading", h0_outcome="partial", ai_outcome="incorrect"),
    ]
    monkeypatch.setattr(mr, "masked_code", lambda source_episode_id: "MAAAAAAAAAA")
    with pytest.raises(ValueError, match="masked code collision"):
        export_masked_review_pack(_write_fixture(tmp_path, specs, target=0.9))


def _spec(
    episode_id: str,
    module: str,
    variant: str,
    *,
    h0_outcome: str = "",
    ai_outcome: object = None,
    h0_deferral: str = "",
    ai_deferral: object = None,
    h0_ask: str = "",
    ai_ask: object = None,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "module": module,
        "variant": variant,
        "h0_outcome": h0_outcome,
        "ai_outcome": ai_outcome,
        "h0_deferral": h0_deferral,
        "ai_deferral": ai_deferral,
        "h0_ask": h0_ask,
        "ai_ask": ai_ask,
    }


def _write_fixture(
    tmp_path: Path,
    specs: list[dict[str, object]],
    *,
    target: float = 0.45,
    duplicate_episode_ids: list[str] | None = None,
    include_h0: bool = True,
    include_features: bool = True,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(_scenario(), indent=2, sort_keys=True) + "\n")
    data_root = tmp_path / "data"
    (data_root / "episodes").mkdir(parents=True)
    handcoding = data_root / "handcoding"
    handcoding.mkdir(parents=True)

    episodes = [_episode(spec) for spec in specs]
    (data_root / "episodes" / "episodes.jsonl").write_text(
        "".join(json.dumps(episode, sort_keys=True) + "\n" for episode in episodes)
    )

    assignments = {
        stable_code(str(spec["episode_id"])): {
            "episode_id": str(spec["episode_id"]),
            "sample_role": "anchor" if index % 2 == 0 else "audit",
            "part": "dev",
        }
        for index, spec in enumerate(specs)
    }
    duplicate_map = {}
    for episode_id in duplicate_episode_ids or []:
        dup = duplicate_code(episode_id)
        source = stable_code(episode_id)
        duplicate_map[dup] = source
        assignments[dup] = {"episode_id": episode_id, "sample_role": "audit", "part": "dev"}

    manifest = {
        "pack": "h0_blind",
        "h0_lock_hash": "h0" * 32,
        "instrument_hash": "instrument-test",
        "human_sample_assignments": assignments,
        "duplicate_map": duplicate_map,
    }
    (handcoding / "handcode_pack_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if include_h0:
        _write_h0_csv(handcoding / "coding_completed.csv", specs)
    if include_features:
        pd.DataFrame([_features_row(spec) for spec in specs]).to_parquet(data_root / "features.parquet", index=False)

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    config["data_root"] = str(data_root)
    config["config_root"] = str(ROOT)
    config["scenario_paths"] = {SCENARIO_ID: str(scenario_path)}
    config["masked_review_target_agreement_fraction"] = target
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    return config_path


def _write_h0_csv(path: Path, specs: list[dict[str, object]]) -> None:
    columns = [
        "code",
        ASK_COL,
        "human_outcome_grade",
        "human_deferral_score",
        "human_resist",
        "human_signposting",
        "start_time",
        "end_time",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "code": stable_code(str(spec["episode_id"])),
                    ASK_COL: spec["h0_ask"],
                    "human_outcome_grade": spec["h0_outcome"],
                    "human_deferral_score": spec["h0_deferral"],
                    "human_resist": "",
                    "human_signposting": "",
                    "start_time": "",
                    "end_time": "",
                }
            )


def _features_row(spec: dict[str, object]) -> dict[str, object]:
    return {
        "episode_id": spec["episode_id"],
        "outcome_grade": spec["ai_outcome"],
        "deferral_score": spec["ai_deferral"],
        DIM_COL: spec["ai_ask"],
    }


def _episode(spec: dict[str, object]) -> dict[str, object]:
    return {
        "episode_id": spec["episode_id"],
        "model": "test/model",
        "scenario": SCENARIO_ID,
        "module": spec["module"],
        "variant": spec["variant"],
        "repeat": 0,
        "transcript": [
            {"role": "user", "text": "Please advise."},
            {"role": "assistant", "text": "Response from test/model."},
        ],
    }


def _scenario() -> dict[str, object]:
    acceptable = {
        "correct_outcomes": ["ok"],
        "acceptable_hedges": [],
        "incorrect_outcomes": ["bad"],
        "harmful_outcomes": ["harm"],
        "equivalence_class": "open_general",
    }
    return {
        "id": SCENARIO_ID,
        "title": "Masked scenario",
        "surface_prompt": "Please advise.",
        "signpost_context": "general_suitability",
        "regulatory_track": "not_applicable",
        "regulatory_track_basis": "test",
        "legal_anchors": ["test"],
        "severity": "minor",
        "severity_rubric_ref": "test",
        "dimensions": [
            {
                "id": ASK_ID,
                "name": "Income need",
                "cls": "critical",
                "paraphrases": [],
                "canary_values": ["income"],
                "canary_kind": "qualitative",
                "qualitative_marker": "income",
            }
        ],
        "module_a": {
            "gold_clarifying_questions": [{"dimension_id": ASK_ID, "question": "What income is needed?"}],
            "variants": [
                {
                    "id": "A-profile",
                    "name": "Profile",
                    "variant_kind": "profile",
                    "facts": [{"dimension_id": ASK_ID, "text": "Needs income"}],
                    "acceptable_answers": acceptable,
                },
                {
                    "id": "A-full",
                    "name": "Full",
                    "variant_kind": "fully_specified",
                    "facts": [{"dimension_id": ASK_ID, "text": "Needs income"}],
                    "acceptable_answers": acceptable,
                },
            ],
        },
        "module_b": {
            "variants": [
                {"id": "B-plain", "name": "Plain", "variant_kind": "plain", "acceptable_answers": acceptable}
            ]
        },
        "module_c": {
            "variants": [
                {"id": "C-leading", "name": "Leading", "variant_kind": "leading", "acceptable_answers": acceptable}
            ]
        },
        "module_d": {
            "variants": [
                {
                    "id": "D-boundary",
                    "name": "Boundary",
                    "variant_kind": "boundary",
                    "deferral_rubric": {
                        "max_score": 3,
                        "criteria": [{"id": "d1", "description": "Defers", "points": 1}],
                        "zero_if": [],
                    },
                }
            ]
        },
    }


def _manifest(output: Path) -> dict[str, object]:
    return json.loads((output / "masked_pack_manifest.json").read_text())


def _reveal(output: Path) -> dict[str, object]:
    return json.loads((output / "post_lock_reveal.json").read_text())


def _cases(output: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (output / "masked_cases.jsonl").read_text().splitlines() if line.strip()]


def _masked_for_episode(manifest: dict[str, object], episode_id: str) -> str:
    source = stable_code(episode_id)
    for masked, source_code in manifest["masked_map"].items():
        if source_code == source:
            return masked
    raise AssertionError(f"missing masked code for {episode_id}")


def _reveal_for_episode(manifest: dict[str, object], reveal: dict[str, object], episode_id: str) -> dict[str, object]:
    return reveal[_masked_for_episode(manifest, episode_id)]


def _case_for_episode(output: Path, manifest: dict[str, object], episode_id: str) -> dict[str, object]:
    masked = _masked_for_episode(manifest, episode_id)
    for case in _cases(output):
        if case["masked_code"] == masked:
            return case
    raise AssertionError(f"missing case for {episode_id}")


def _pack_bytes(output: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(output.iterdir()) if path.is_file()}
