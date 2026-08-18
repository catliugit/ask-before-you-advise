from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import yaml

from slice.handcode import (
    DEFAULT_HANDCODE_DUPLICATE_SEED_OFFSET,
    DEFAULT_HANDCODE_SEED,
    DEFAULT_HANDCODE_SIZES,
    _allocate_anchor_tasks,
    _allocate_danger_cap,
    _danger_band,
    _duplicate_row,
    _h0_lock_hash,
    _load_scenarios,
    _select_duplicate_source_ids,
    duplicate_code,
    export_handcode_pack,
    sample_danger_zone,
    sample_for_handcode,
    sample_gold_set,
    sample_representative_anchor,
    stable_code,
)
from slice.schema import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_sample_for_handcode_is_stratified_and_hits_prereg_module_sizes():
    assert DEFAULT_HANDCODE_SIZES == {"A": 30, "B": 30, "C": 50, "D": 50}

    episodes = []
    for index in range(40):
        episodes.append(_episode(f"A-{index}", module="A", variant_kind="profile" if index % 2 else "fully_specified"))
        episodes.append(_episode(f"B-{index}", module="B", variant_kind="leading" if index % 2 else "plain"))
    for index in range(60):
        episodes.append(_episode(f"C-{index}", module="C", variant_kind=["control", "disclosed", "placebo"][index % 3]))
    for index in range(3):
        episodes.append(_episode(f"D-{index}", module="D", variant_kind="boundary"))
    for index in range(10):
        episodes.append(_episode(f"D-marker-{index}", module="D", variant_kind="boundary", calibration_gate=True))

    sampled_ids = sample_for_handcode(episodes, {"A": 30, "B": 10, "C": 50, "D": 50}, seed=123)
    by_id = {episode["episode_id"]: episode for episode in episodes}
    counts = Counter(by_id[episode_id]["module"] for episode_id in sampled_ids)

    assert counts == {"A": 30, "B": 30, "C": 50, "D": 3}
    assert all(not by_id[episode_id].get("calibration_gate", False) for episode_id in sampled_ids)
    assert {by_id[episode_id]["variant_kind"] for episode_id in sampled_ids if by_id[episode_id]["module"] == "C"} == {
        "control",
        "disclosed",
        "placebo",
    }


def test_export_handcode_pack_is_blind_shuffled_and_uses_codebook_axis_columns(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml")
    data_root = tmp_path / "data"
    episodes = [
        _episode("S1-A-A1-r0", module="A", scenario="S1", variant="A1"),
        _episode("S1-B-B-neutral-r0", module="B", scenario="S1", variant="B-neutral"),
        _episode("S1-C-C-control-r0", module="C", scenario="S1", variant="C-control"),
        _episode("B-scam-D-D-neutral-r0", module="D", scenario="B-scam", variant="D-neutral"),
        _episode("S1-B-B-neutral-marker", module="B", scenario="S1", variant="B-neutral", calibration_gate=True),
    ]
    _write_jsonl(data_root / "episodes" / "episodes.jsonl", episodes)

    handcode_dir = export_handcode_pack(config_path)
    template = (handcode_dir / "coding_template.csv").read_text()
    transcripts = [json.loads(line) for line in (handcode_dir / "transcripts.jsonl").read_text().splitlines()]
    manifest = json.loads((handcode_dir / "handcode_pack_manifest.json").read_text())

    assert "stub/test" not in template
    assert "stub/test" not in json.dumps(transcripts)
    assert "test_model" not in json.dumps(transcripts)
    assert "human_dim_" not in template
    assert "human_ask_s1.d1" in template
    assert "human_outcome_grade" in template
    assert "human_deferral_score" in template
    assert "human_resist" in template
    assert "human_signposting" in template
    assert manifest["seed"]
    assert manifest["dev_test_split_ratio"] == 0.7
    assert manifest["target_n"] == 250
    assert manifest["effective_target_n"] == 4
    assert manifest["anchor_requested_n"] == 150
    assert manifest["anchor_drawn_n"] == 4
    assert manifest["audit_drawn_n"] == 0
    assert manifest["sample_role_counts"] == {"anchor": 4, "audit": 0}
    assert "per_stratum_allocation" not in manifest
    assert "danger_band_counts" not in manifest
    assert manifest["frames"]["anchor"]["per_danger_band"] == {
        "boundary": {"available": 1, "drawn": 1},
        "cheap_fine_on_safety": {"available": 0, "drawn": 0},
        "standard": {"available": 3, "drawn": 3},
    }
    assert manifest["frames"]["anchor"]["per_task"]["D"] == {"available": 1, "drawn": 1, "floor": 1}
    assert manifest["frames"]["audit"]["per_stratum_allocation"] == {}
    assert manifest["frames"]["audit"]["danger_band_counts"] == {
        "boundary": 0,
        "cheap_fine_on_safety": 0,
        "standard": 0,
    }
    assert manifest["frames"]["audit"]["stratification_weights"] == {
        "boundary": 3.0,
        "cheap_fine_on_safety": 4.0,
        "standard": 1.0,
    }
    assert manifest["frames"]["audit"]["cheap_consensus_available"] is False
    assert stable_code("S1-B-B-neutral-marker") not in manifest["human_sample_assignments"]
    assert {assignment["sample_role"] for assignment in manifest["human_sample_assignments"].values()} == {"anchor"}
    assert manifest["duplicate_map"] == {}
    assert manifest["n_duplicates"] == 0

    codes = [row["code"] for row in transcripts]
    assert codes == _expected_pack_order(codes, manifest)

    rows = list(csv.DictReader((handcode_dir / "coding_template.csv").open()))
    assert all("stub/test" not in json.dumps(row) for row in rows)


def test_export_handcode_pack_duplicate_ordering_resolves_all_codes(tmp_path):
    handcode_dir, manifest, transcripts = _export_duplicate_fixture(tmp_path / "ordered", duplicate_fraction=0.30)

    codes = [row["code"] for row in transcripts]

    assert manifest["n_duplicates"] == 3
    assert set(manifest["duplicate_map"]).issubset(codes)
    assert set(manifest["duplicate_map"].values()).issubset(codes)
    assert codes == _expected_pack_order(codes, manifest)
    assert {path.stem for path in handcode_dir.glob("T*.md")} == set(codes)


def test_duplicate_fraction_count_and_seeded_source_codes_are_deterministic(tmp_path):
    _, manifest, transcripts = _export_duplicate_fixture(tmp_path / "nonzero", duplicate_fraction=0.30)
    sampled_ids = sorted(value["episode_id"] for value in manifest["human_sample_assignments"].values())
    expected_source_ids = [
        "dup-fixture-standard-0",
        "dup-fixture-standard-1",
        "dup-fixture-standard-5",
    ]
    expected_duplicate_map = {
        duplicate_code(episode_id): stable_code(episode_id)
        for episode_id in expected_source_ids
    }

    assert len(manifest["human_sample_assignments"]) == 10
    assert len(transcripts) == 13
    assert manifest["n_duplicates"] == 3
    assert len(manifest["duplicate_map"]) == 3
    assert manifest["duplicate_map"] == expected_duplicate_map
    assert _select_duplicate_source_ids(
        sampled_ids,
        fraction=0.30,
        seed=DEFAULT_HANDCODE_SEED + DEFAULT_HANDCODE_DUPLICATE_SEED_OFFSET,
    ) == expected_source_ids

    _, zero_manifest, zero_transcripts = _export_duplicate_fixture(tmp_path / "zero", duplicate_fraction=0.0)
    assert zero_manifest["duplicate_map"] == {}
    assert zero_manifest["n_duplicates"] == 0
    assert len(zero_transcripts) == 10


def test_duplicate_codes_are_distinct_blinded_t_namespaced_codes(tmp_path):
    _, manifest, _ = _export_duplicate_fixture(tmp_path / "codes", duplicate_fraction=0.30)
    sampled_episode_ids = {
        assignment["episode_id"]
        for assignment in manifest["human_sample_assignments"].values()
    }
    sampled_stable_codes = {stable_code(episode_id) for episode_id in sampled_episode_ids}

    for dup_code, source_code in manifest["duplicate_map"].items():
        assert re.fullmatch(r"T[0-9A-F]{10}", dup_code)
        assert dup_code != source_code
        assert source_code in sampled_stable_codes
        assert dup_code not in sampled_stable_codes


def test_duplicate_transcript_content_matches_source(tmp_path):
    handcode_dir, manifest, transcripts = _export_duplicate_fixture(tmp_path / "content", duplicate_fraction=0.30)
    rows_by_code = {row["code"]: row for row in transcripts}

    for dup_code, source_code in manifest["duplicate_map"].items():
        assert rows_by_code[dup_code]["transcript"] == rows_by_code[source_code]["transcript"]
        assert _md_transcript_body(handcode_dir / f"{dup_code}.md") == _md_transcript_body(
            handcode_dir / f"{source_code}.md"
        )


def test_duplicate_row_preserves_part_and_deep_copies_transcript():
    source = {
        "code": stable_code("source-episode"),
        "episode_id": "source-episode",
        "order_key": hashlib.sha256(b"order:source-episode").hexdigest(),
        "module": "C",
        "scenario": "S1",
        "variant": "C-control",
        "repeat": 0,
        "human_sample": "test",
        "transcript": [
            {"speaker": "user", "text": "Question"},
            {"speaker": "assistant", "text": "Answer"},
        ],
    }

    dup = _duplicate_row(source)
    unchanged_keys = set(source) - {"code", "order_key", "transcript"}

    assert dup["human_sample"] == "test"
    assert {key: dup[key] for key in unchanged_keys} == {key: source[key] for key in unchanged_keys}
    assert dup["code"] == duplicate_code("source-episode")
    assert dup["order_key"] == hashlib.sha256(f"order:{dup['code']}".encode("utf-8")).hexdigest()
    assert dup["transcript"] == source["transcript"]
    assert dup["transcript"] is not source["transcript"]
    assert all(dup_turn is not source_turn for dup_turn, source_turn in zip(dup["transcript"], source["transcript"]))


def test_duplicate_blinding_no_leaks_or_episode_ids_in_coder_files(tmp_path):
    handcode_dir, manifest, transcripts = _export_duplicate_fixture(tmp_path / "blind", duplicate_fraction=0.30)
    rows_by_code = {row["code"]: row for row in transcripts}
    coder_texts = [
        (handcode_dir / "transcripts.jsonl").read_text(),
        (handcode_dir / "coding_template.csv").read_text(),
        (handcode_dir / "instructions.md").read_text(),
        *[path.read_text() for path in sorted(handcode_dir.glob("T*.md"))],
    ]

    for token in ("duplicate", "h0_blind", "h0_lock_hash", "duplicate_map"):
        assert all(token not in text for text in coder_texts)

    for dup_code, source_code in manifest["duplicate_map"].items():
        source_episode_id = manifest["human_sample_assignments"][source_code]["episode_id"]
        assert source_episode_id not in json.dumps(rows_by_code[dup_code], sort_keys=True)
        assert source_episode_id not in (handcode_dir / f"{dup_code}.md").read_text()

    assert manifest["pack"] == "h0_blind"
    assert "h0_lock_hash" in manifest
    assert "duplicate_map" in manifest


def test_h0_lock_hash_present_deterministic_and_content_sensitive(tmp_path):
    handcode_dir, manifest, _ = _export_duplicate_fixture(
        tmp_path / "lock",
        duplicate_fraction=0.30,
        prefix="lock-fixture",
    )
    first_bytes = _coder_file_bytes(handcode_dir)
    repeat_dir, repeat_manifest, _ = _export_duplicate_fixture(
        tmp_path / "lock",
        duplicate_fraction=0.30,
        prefix="lock-fixture",
    )
    _, changed_manifest, _ = _export_duplicate_fixture(
        tmp_path / "lock-changed",
        duplicate_fraction=0.30,
        prefix="lock-fixture",
        transcript_suffix="changed",
    )

    assert re.fullmatch(r"[0-9a-f]{64}", manifest["h0_lock_hash"])
    assert manifest["pack"] == "h0_blind"
    assert manifest["h0_lock_hash"] == repeat_manifest["h0_lock_hash"]
    assert first_bytes == _coder_file_bytes(repeat_dir)
    assert manifest["h0_lock_hash"] == _computed_h0_lock_hash(handcode_dir)
    assert changed_manifest["h0_lock_hash"] != manifest["h0_lock_hash"]


def test_h0_lock_hash_ignores_non_pack_files_in_shared_dir(tmp_path):
    # data/handcoding is shared with council outputs and the human's coding_completed.csv. The H0 lock
    # must cover ONLY coder-facing pack files, or it would break spuriously when those other artefacts
    # change. Drop stray non-pack files into the dir and assert the lock is unchanged.
    handcode_dir, manifest, _ = _export_duplicate_fixture(
        tmp_path / "shared",
        duplicate_fraction=0.30,
        prefix="shared-fixture",
    )
    locked = manifest["h0_lock_hash"]
    assert _h0_lock_hash(handcode_dir) == locked

    (handcode_dir / "council_labels.csv").write_text("code,field,council_label\nT123,outcome,correct\n")
    (handcode_dir / "coding_completed.csv").write_text("code,human_outcome_grade\nT123,correct\n")
    (handcode_dir / "council_pre_deliberation.csv").write_text("code,coder,label\nT123,stub,correct\n")

    assert _h0_lock_hash(handcode_dir) == locked
    assert _computed_h0_lock_hash(handcode_dir) == locked


def test_stale_transcript_files_are_removed_on_reexport(tmp_path):
    handcode_dir, first_manifest, _ = _export_duplicate_fixture(tmp_path / "stale", duplicate_fraction=0.30)
    stale_duplicate_files = {handcode_dir / f"{code}.md" for code in first_manifest["duplicate_map"]}
    assert stale_duplicate_files
    assert all(path.exists() for path in stale_duplicate_files)

    handcode_dir, manifest, transcripts = _export_duplicate_fixture(tmp_path / "stale", duplicate_fraction=0.0)
    current_codes = {row["code"] for row in transcripts}
    disk_codes = {path.stem for path in handcode_dir.glob("T*.md")}

    assert manifest["duplicate_map"] == {}
    assert manifest["n_duplicates"] == 0
    assert disk_codes == current_codes
    assert all(not path.exists() for path in stale_duplicate_files)
    assert manifest["h0_lock_hash"] == _computed_h0_lock_hash(handcode_dir)


def test_whole_handcode_pack_is_deterministic_with_duplicates(tmp_path):
    first_dir, first_manifest, _ = _export_duplicate_fixture(
        tmp_path / "determinism-a",
        duplicate_fraction=0.30,
        prefix="determinism-fixture",
    )
    second_dir, second_manifest, _ = _export_duplicate_fixture(
        tmp_path / "determinism-b",
        duplicate_fraction=0.30,
        prefix="determinism-fixture",
    )

    assert first_manifest["duplicate_map"] == second_manifest["duplicate_map"]
    assert first_manifest["h0_lock_hash"] == second_manifest["h0_lock_hash"]
    assert _coder_file_bytes(first_dir) == _coder_file_bytes(second_dir)


def test_gold_set_frames_disjoint_and_role_counts(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml")
    scenarios = _load_scenarios(load_config(config_path))
    episodes = [
        _boundary_episode(f"D-boundary-{index}", variant="D-neutral")
        for index in range(3)
    ]
    episodes.extend(
        [
            _episode("B-harmful-0", module="B", scenario="S1", variant="B-neutral"),
            *_standard_episodes("gold-roles", 12),
        ]
    )

    result = sample_gold_set(
        episodes,
        scenarios,
        target_n=8,
        anchor_n=3,
        task_floors={"A": 0, "B": 1, "C": 1, "D": 1},
        per_model_floor=2,
        danger_cap_fraction=1.0,
        weights={},
        cheap_outcomes={"B-harmful-0": "harmful"},
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )

    anchor_ids = {episode_id for episode_id, role in result.roles.items() if role == "anchor"}
    audit_ids = {episode_id for episode_id, role in result.roles.items() if role == "audit"}
    assert anchor_ids.isdisjoint(audit_ids)
    assert {stable_code(episode_id) for episode_id in anchor_ids}.isdisjoint(
        {stable_code(episode_id) for episode_id in audit_ids}
    )
    assert set(result.sampled_ids) == set(result.roles)
    assert sum(result.accounting["sample_role_counts"].values()) == len(result.sampled_ids)
    assert result.accounting["sample_role_counts"]["anchor"] == len(anchor_ids)
    assert result.accounting["sample_role_counts"]["audit"] == len(audit_ids)
    assert sum(
        item["drawn"] for item in result.accounting["frames"]["anchor"]["per_task"].values()
    ) == len(anchor_ids)


def test_anchor_is_grade_proportional_within_task(tmp_path):
    # Cell layout is deliberately ANTI-correlated with the grade proportion so the two algorithms diverge:
    # the majority grade (80 `correct`) sits in ONE model cell, the minority (20 `harmful`) is spread over
    # FOUR cells. Proportional-by-grade draws 16/4 (the population ratio). The old round-robin-across-cells
    # would pop ~evenly across the 5 cells -> ~4 correct / ~16 harmful, the OPPOSITE. Asserting 16/4 catches
    # it. (A fixture whose cell ratio equals the grade ratio would pass under BOTH algorithms and prove
    # nothing.)
    config_path = _write_config(tmp_path / "config.yaml")
    scenarios = _load_scenarios(load_config(config_path))
    episodes = []
    cheap_outcomes = {}
    for item_index in range(80):
        episode_id = f"C-correct-{item_index}"
        episode = _episode(episode_id, module="C", scenario="S1", variant="C-control")
        episode["model"] = "correct-model"
        episodes.append(episode)
        cheap_outcomes[episode_id] = "correct"
    for model_index in range(4):
        for item_index in range(5):
            episode_id = f"C-harmful-{model_index}-{item_index}"
            episode = _episode(episode_id, module="C", scenario="S1", variant="C-control")
            episode["model"] = f"harmful-model-{model_index}"
            episodes.append(episode)
            cheap_outcomes[episode_id] = "harmful"

    result = sample_representative_anchor(
        episodes,
        scenarios,
        anchor_n=20,
        task_floors={"C": 0},
        per_model_floor=2,
        cheap_outcomes=cheap_outcomes,
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )
    drawn_grades = Counter(cheap_outcomes[episode_id] for episode_id in result.ids)

    assert drawn_grades == {"correct": 16, "harmful": 4}
    assert result.accounting["per_task_grade"]["C:correct"] == {"available": 80, "drawn": 16}
    assert result.accounting["per_task_grade"]["C:harmful"] == {"available": 20, "drawn": 4}


def test_anchor_task_floors_boundary_protected_exact():
    assert _allocate_anchor_tasks(
        {"A": 80, "B": 80, "C": 80, "D": 20},
        30,
        {"A": 10, "B": 10, "C": 10, "D": 20},
    ) == {"A": 4, "B": 3, "C": 3, "D": 20}


def test_anchor_floors_fit_surplus_is_population_proportional():
    allocation = _allocate_anchor_tasks(
        {"A": 100, "B": 50, "C": 10, "D": 20},
        60,
        {"A": 5, "B": 5, "C": 5, "D": 5},
    )

    assert allocation == {"A": 29, "B": 16, "C": 6, "D": 9}
    assert allocation["A"] - 5 > allocation["B"] - 5 > allocation["D"] - 5 > allocation["C"] - 5


def test_audit_targeted_disjoint_and_residual_recorded(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml")
    scenarios = _load_scenarios(load_config(config_path))
    episodes = [
        *[_boundary_episode(f"D-boundary-{index}", variant="D-neutral") for index in range(4)],
        _boundary_episode("D-cleared-0", variant="D-urgent"),
        *[
            _episode(f"B-harmful-{index}", module="B", scenario="S1", variant="B-neutral")
            for index in range(3)
        ],
        *_standard_episodes("residual", 20),
    ]
    cheap_outcomes = {
        "D-cleared-0": "correct",
        **{f"B-harmful-{index}": "harmful" for index in range(3)},
    }

    result = sample_gold_set(
        episodes,
        scenarios,
        target_n=8,
        anchor_n=4,
        task_floors={"A": 0, "B": 0, "C": 0, "D": 4},
        per_model_floor=2,
        danger_cap_fraction=1.0,
        weights={},
        cheap_outcomes=cheap_outcomes,
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )

    anchor_ids = {episode_id for episode_id, role in result.roles.items() if role == "anchor"}
    audit_ids = {episode_id for episode_id, role in result.roles.items() if role == "audit"}
    anchor = result.accounting["frames"]["anchor"]
    audit = result.accounting["frames"]["audit"]

    assert anchor_ids.isdisjoint(audit_ids)
    assert anchor["per_task"]["D"]["drawn"] == 4
    assert anchor["per_danger_band"]["boundary"]["drawn"] == 3
    assert anchor["per_danger_band"]["cheap_fine_on_safety"]["drawn"] == 1
    assert audit["danger_band_counts"] == {"boundary": 4, "cheap_fine_on_safety": 0, "standard": 0}
    assert audit["per_stratum_allocation"]["D:boundary:boundary"] == {
        "available": 1,
        "band": "boundary",
        "drawn": 1,
    }
    assert audit["per_stratum_allocation"]["B:plain:boundary"] == {
        "available": 3,
        "band": "boundary",
        "drawn": 3,
    }


def test_sample_role_not_visible_to_coder(tmp_path):
    config_path = _write_config(
        tmp_path / "config.yaml",
        target_n=5,
        anchor_n=2,
        anchor_task_floors={"A": 0, "B": 0, "C": 0, "D": 2},
        danger_cap_fraction=1.0,
    )
    data_root = tmp_path / "data"
    episodes = [
        _boundary_episode("D-boundary-0", variant="D-neutral"),
        _boundary_episode("D-cleared-0", variant="D-urgent"),
        _episode("B-harmful-0", module="B", scenario="S1", variant="B-neutral"),
        *_standard_episodes("blind", 4),
    ]
    _write_jsonl(data_root / "episodes" / "episodes.jsonl", episodes)
    _write_cheap_panel_judgements(
        data_root / "judgements.jsonl",
        [
            {"episode_id": "D-cleared-0", "outcome": "correct"},
            {"episode_id": "B-harmful-0", "outcome": "harmful"},
        ],
    )

    handcode_dir = export_handcode_pack(config_path)
    manifest = json.loads((handcode_dir / "handcode_pack_manifest.json").read_text())
    public_texts = [
        (handcode_dir / "transcripts.jsonl").read_text(),
        (handcode_dir / "coding_template.csv").read_text(),
        *[path.read_text() for path in sorted(handcode_dir.glob("T*.md"))],
    ]

    assert {assignment["sample_role"] for assignment in manifest["human_sample_assignments"].values()} == {
        "anchor",
        "audit",
    }
    assert "anchor" in json.dumps(manifest)
    assert "audit" in json.dumps(manifest)
    assert all("anchor" not in text for text in public_texts)
    assert all("audit" not in text for text in public_texts)


def test_anchor_under_or_over_budget_terminates(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml")
    scenarios = _load_scenarios(load_config(config_path))
    underfilled_episodes = [
        _boundary_episode("D-under", variant="D-neutral"),
        *_standard_episodes("under", 2),
    ]
    underfilled = sample_gold_set(
        underfilled_episodes,
        scenarios,
        target_n=10,
        anchor_n=10,
        task_floors={"A": 0, "B": 0, "C": 0, "D": 10},
        per_model_floor=2,
        danger_cap_fraction=0.7,
        weights={},
        cheap_outcomes={},
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )

    assert underfilled.sampled_ids == sorted(episode["episode_id"] for episode in underfilled_episodes)
    assert underfilled.accounting["anchor_drawn_n"] == 3
    assert underfilled.accounting["audit_drawn_n"] == 0
    assert underfilled.accounting["effective_target_n"] == 3

    over_budget_episodes = _standard_episodes("over", 10)
    over_budget = sample_gold_set(
        over_budget_episodes,
        scenarios,
        target_n=4,
        anchor_n=20,
        task_floors={"A": 0, "B": 10, "C": 10, "D": 0},
        per_model_floor=2,
        danger_cap_fraction=0.7,
        weights={},
        cheap_outcomes={},
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )
    over_budget_repeat = sample_gold_set(
        over_budget_episodes,
        scenarios,
        target_n=4,
        anchor_n=20,
        task_floors={"A": 0, "B": 10, "C": 10, "D": 0},
        per_model_floor=2,
        danger_cap_fraction=0.7,
        weights={},
        cheap_outcomes={},
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )

    assert over_budget.accounting["anchor_requested_n"] == 4
    assert over_budget.accounting["anchor_drawn_n"] == 4
    assert over_budget.accounting["audit_drawn_n"] == 0
    assert over_budget.sampled_ids == over_budget_repeat.sampled_ids
    assert over_budget.roles == over_budget_repeat.roles


def test_danger_first_includes_whole_danger_zone(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml", target_n=10)
    scenarios = _load_scenarios(load_config(config_path))
    danger_ids = {
        "D-boundary-0",
        "D-boundary-1",
        "D-cleared-0",
        "D-cleared-1",
        "B-harmful-0",
    }
    episodes = [
        _boundary_episode("D-boundary-0", variant="D-neutral"),
        _boundary_episode("D-boundary-1", variant="D-urgent"),
        _boundary_episode("D-cleared-0", variant="D-neutral"),
        _boundary_episode("D-cleared-1", variant="D-urgent"),
        _episode("B-harmful-0", module="B", scenario="S1", variant="B-neutral"),
        *_standard_episodes("danger-first", 20),
    ]
    result = sample_danger_zone(
        episodes,
        scenarios,
        target_n=10,
        danger_cap_fraction=0.7,
        weights={},
        cheap_outcomes={
            "D-cleared-0": "correct",
            "D-cleared-1": "correct",
            "B-harmful-0": "harmful",
        },
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )

    sampled_ids = result.sampled_ids

    assert danger_ids <= set(sampled_ids)
    assert "B-harmful-0" in sampled_ids
    assert len(sampled_ids) == 10
    assert result.accounting["effective_target_n"] == 10
    assert result.accounting["danger_band_counts"] == {"boundary": 3, "cheap_fine_on_safety": 2, "standard": 5}
    assert result.accounting["per_stratum_allocation"]["D:boundary:boundary"]["drawn"] == 2
    assert result.accounting["per_stratum_allocation"]["D:boundary:cheap_fine_on_safety"]["drawn"] == 2
    assert result.accounting["per_stratum_allocation"]["B:plain:boundary"]["drawn"] == 1


def test_harm_flagged_non_d_episode_is_boundary_danger_band(tmp_path):
    non_d = _episode("B-flag-only-harm", module="B", scenario="S1", variant="B-neutral")
    cleared_d = _boundary_episode("D-cleared-flagged", variant="D-neutral")

    assert _danger_band(non_d, "correct", harm_flagged=True) == "boundary"
    assert _danger_band(non_d, "correct", harm_flagged=False) == "standard"
    assert _danger_band(cleared_d, "correct", harm_flagged=True) == "cheap_fine_on_safety"
    assert _danger_band(cleared_d, "correct", harm_flagged=False) == "cheap_fine_on_safety"


def test_sample_danger_zone_places_flag_only_harm_episode_in_danger_zone(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml")
    scenarios = _load_scenarios(load_config(config_path))
    flagged_id = "B-flag-only-harm"
    episodes = [
        _episode(flagged_id, module="B", scenario="S1", variant="B-neutral"),
        *_standard_episodes("flag-only-harm", 4),
    ]

    result = sample_danger_zone(
        episodes,
        scenarios,
        target_n=1,
        danger_cap_fraction=1.0,
        weights={},
        cheap_outcomes={flagged_id: "correct"},
        harm_flagged_ids={flagged_id},
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )

    assert result.sampled_ids == [flagged_id]
    assert result.accounting["danger_band_counts"] == {
        "boundary": 1,
        "cheap_fine_on_safety": 0,
        "standard": 0,
    }
    assert result.accounting["per_stratum_allocation"]["B:plain:boundary"]["drawn"] == 1


def test_cheap_fine_on_safety_band_requires_two_agreeing_cheap_rows(tmp_path):
    config_path = _write_config(
        tmp_path / "config.yaml",
        target_n=4,
        anchor_n=1,
        anchor_task_floors={"A": 0, "B": 0, "C": 0, "D": 0},
        danger_cap_fraction=1.0,
    )
    data_root = tmp_path / "data"
    episodes = [
        _boundary_episode("D-one-row", variant="D-neutral"),
        _boundary_episode("D-two-rows", variant="D-urgent"),
        *_standard_episodes("cheap-consensus", 2),
    ]
    _write_jsonl(data_root / "episodes" / "episodes.jsonl", episodes)
    _write_cheap_panel_judgements(
        data_root / "judgements.jsonl",
        [
            {"episode_id": "D-one-row", "outcome": "correct", "copies": 1},
            {"episode_id": "D-two-rows", "outcome": "correct"},
        ],
    )

    manifest = _export_manifest(config_path)

    audit = manifest["frames"]["audit"]
    assert manifest["anchor_drawn_n"] == 1
    assert manifest["audit_drawn_n"] == 3
    assert audit["danger_band_counts"] == {"boundary": 1, "cheap_fine_on_safety": 1, "standard": 1}
    assert audit["per_stratum_allocation"]["D:boundary:boundary"]["drawn"] == 1
    assert audit["per_stratum_allocation"]["D:boundary:cheap_fine_on_safety"]["drawn"] == 1


def test_gold_set_deterministic(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml", target_n=12)
    scenarios = _load_scenarios(load_config(config_path))
    episodes = [
        _boundary_episode("D-boundary-0", variant="D-neutral"),
        _boundary_episode("D-cleared-0", variant="D-urgent"),
        _episode("B-harmful-0", module="B", scenario="S1", variant="B-neutral"),
        *_standard_episodes("deterministic", 30),
    ]
    cheap_outcomes = {
        "D-cleared-0": "partial",
        "B-harmful-0": "harmful",
    }

    result_one = sample_gold_set(
        episodes,
        scenarios,
        target_n=12,
        anchor_n=5,
        task_floors={"A": 0, "B": 1, "C": 1, "D": 2},
        per_model_floor=2,
        danger_cap_fraction=0.7,
        weights={},
        cheap_outcomes=cheap_outcomes,
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )
    result_two = sample_gold_set(
        episodes,
        scenarios,
        target_n=12,
        anchor_n=5,
        task_floors={"A": 0, "B": 1, "C": 1, "D": 2},
        per_model_floor=2,
        danger_cap_fraction=0.7,
        weights={},
        cheap_outcomes=cheap_outcomes,
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )

    assert result_one.sampled_ids == result_two.sampled_ids
    assert result_one.roles == result_two.roles
    assert result_one.accounting["frames"] == result_two.accounting["frames"]


def test_danger_first_without_judgements_falls_back(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml", target_n=3)
    data_root = tmp_path / "data"
    episodes = [
        _boundary_episode("D-no-judgements", variant="D-neutral"),
        *_standard_episodes("fallback", 3),
    ]
    _write_jsonl(data_root / "episodes" / "episodes.jsonl", episodes)

    handcode_dir = export_handcode_pack(config_path)
    manifest = json.loads((handcode_dir / "handcode_pack_manifest.json").read_text())

    assert (handcode_dir / "transcripts.jsonl").exists()
    assert manifest["frames"]["audit"]["cheap_consensus_available"] is False
    assert manifest["frames"]["anchor"]["per_danger_band"] == {
        "boundary": {"available": 1, "drawn": 1},
        "cheap_fine_on_safety": {"available": 0, "drawn": 0},
        "standard": {"available": 3, "drawn": 2},
    }


def test_danger_cap_fraction_limits_danger_share(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml", target_n=10, danger_cap_fraction=0.5)
    scenarios = _load_scenarios(load_config(config_path))
    plain_danger = [_boundary_episode(f"D-boundary-{index}", variant="D-neutral") for index in range(10)]
    cleared_danger = [_boundary_episode(f"D-cleared-{index}", variant="D-urgent") for index in range(10)]
    episodes = [*plain_danger, *cleared_danger, *_standard_episodes("cap", 20)]
    result = sample_danger_zone(
        episodes,
        scenarios,
        target_n=10,
        danger_cap_fraction=0.5,
        weights={},
        cheap_outcomes={episode["episode_id"]: "correct" for episode in cleared_danger},
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )

    assert result.accounting["danger_band_counts"] == {"boundary": 2, "cheap_fine_on_safety": 3, "standard": 5}
    assert result.accounting["per_stratum_allocation"]["D:boundary:boundary"]["drawn"] == 2
    assert result.accounting["per_stratum_allocation"]["D:boundary:cheap_fine_on_safety"]["drawn"] == 3


def test_allocation_reaches_target_when_standard_is_insufficient(tmp_path):
    config_path = _write_config(tmp_path / "binding" / "config.yaml", target_n=100, danger_cap_fraction=0.7)
    scenarios = _load_scenarios(load_config(config_path))
    episodes = [
        *[_boundary_episode(f"D-large-{index}", variant="D-neutral") for index in range(240)],
        *_standard_episodes("insufficient-standard", 10),
    ]
    result = sample_danger_zone(
        episodes,
        scenarios,
        target_n=100,
        danger_cap_fraction=0.7,
        weights={},
        cheap_outcomes={},
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )

    assert len(result.sampled_ids) == result.accounting["effective_target_n"] == 100
    assert result.accounting["danger_band_counts"]["boundary"] == 90
    assert result.accounting["danger_band_counts"]["standard"] == 10
    assert result.accounting["danger_band_counts"]["boundary"] != 70

    small_config_path = _write_config(tmp_path / "small" / "config.yaml", target_n=250, danger_cap_fraction=0.7)
    small_scenarios = _load_scenarios(load_config(small_config_path))
    small_episodes = [
        *[_boundary_episode(f"D-small-{index}", variant="D-neutral") for index in range(90)],
        *_standard_episodes("small-total", 10),
    ]
    small_result = sample_danger_zone(
        small_episodes,
        small_scenarios,
        target_n=250,
        danger_cap_fraction=0.7,
        weights={},
        cheap_outcomes={},
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )
    available = sum(item["available"] for item in small_result.accounting["per_stratum_allocation"].values())

    assert available == 100
    assert available < small_result.accounting["target_n"]
    assert small_result.accounting["effective_target_n"] == available
    assert len(small_result.sampled_ids) == available


def test_manifest_allocation_sums_are_consistent(tmp_path):
    config_path = _write_config(
        tmp_path / "config.yaml",
        target_n=9,
        anchor_n=4,
        anchor_task_floors={"A": 0, "B": 1, "C": 1, "D": 2},
        danger_cap_fraction=1.0,
    )
    data_root = tmp_path / "data"
    episodes = [
        _boundary_episode("D-boundary-0", variant="D-neutral"),
        _boundary_episode("D-cleared-0", variant="D-urgent"),
        _episode("B-harmful-0", module="B", scenario="S1", variant="B-neutral"),
        *_standard_episodes("invariant", 12),
    ]
    _write_jsonl(data_root / "episodes" / "episodes.jsonl", episodes)
    _write_cheap_panel_judgements(
        data_root / "judgements.jsonl",
        [
            {"episode_id": "D-cleared-0", "outcome": "correct"},
            {"episode_id": "B-harmful-0", "outcome": "harmful"},
        ],
    )

    manifest = _export_manifest(config_path)
    anchor_task_sum = sum(item["drawn"] for item in manifest["frames"]["anchor"]["per_task"].values())
    audit_drawn_sum = sum(item["drawn"] for item in manifest["frames"]["audit"]["per_stratum_allocation"].values())
    audit_band_sum = sum(manifest["frames"]["audit"]["danger_band_counts"].values())
    sampled_count = len(_sampled_ids_from_manifest(manifest))
    assignment_count = len(manifest["human_sample_assignments"])
    role_count = sum(manifest["sample_role_counts"].values())

    assert manifest["n_duplicates"] == 1
    assert len(manifest["duplicate_map"]) == 1
    assert set(manifest["duplicate_map"]).isdisjoint(manifest["human_sample_assignments"])
    assert anchor_task_sum == manifest["anchor_drawn_n"]
    assert audit_drawn_sum == audit_band_sum == manifest["audit_drawn_n"]
    assert role_count == sampled_count == assignment_count == manifest["effective_target_n"]
    assert manifest["anchor_drawn_n"] + manifest["audit_drawn_n"] == manifest["effective_target_n"]


def test_sample_danger_zone_returns_canonically_sorted_ids(tmp_path):
    # The dev/test split (via _assign_human_sample_parts) is order-sensitive, so the helper MUST return
    # ids in canonical sorted order, not merely a stable arbitrary order.
    config_path = _write_config(tmp_path / "config.yaml")
    scenarios = _load_scenarios(load_config(config_path))
    episodes = [
        _episode(f"S1-C-C-control-{i}", module="C", scenario="S1", variant="C-control", variant_kind="control")
        for i in range(12)
    ]

    result = sample_danger_zone(
        episodes,
        scenarios,
        target_n=8,
        danger_cap_fraction=0.7,
        weights={},
        cheap_outcomes={},
        seed=DEFAULT_HANDCODE_SEED,
        split="development",
    )

    assert len(result.sampled_ids) == 8
    assert result.sampled_ids == sorted(result.sampled_ids)


def test_allocate_danger_cap_zero_weights_and_dynamic_remainder():
    # Zero weights -> availability-proportional fallback (not a degenerate empty/lexicographic fill).
    equal = _allocate_danger_cap(
        {("D", "boundary", "boundary"): 4, ("D", "boundary", "cheap_fine_on_safety"): 4},
        4,
        {"boundary": 0.0, "cheap_fine_on_safety": 0.0},
    )
    assert sum(equal.values()) == 4
    assert equal[("D", "boundary", "boundary")] == 2
    assert equal[("D", "boundary", "cheap_fine_on_safety")] == 2

    # Dynamic remainder SPREADS the leftover seats across strata. Three equal, uncapped strata sharing a
    # budget of 5 must come out [2, 2, 1] (max 2); a STATIC-remainder allocator would hand both leftovers
    # to the single largest-fraction stratum -> [3, 1, 1] (max 3). This is the starvation the review caught.
    spread = _allocate_danger_cap(
        {
            ("D", "v1", "boundary"): 10,
            ("D", "v2", "boundary"): 10,
            ("D", "v3", "boundary"): 10,
        },
        5,
        {"boundary": 1.0},
    )
    assert sum(spread.values()) == 5
    assert sorted(spread.values()) == [1, 2, 2]
    assert max(spread.values()) == 2

    # A heavily-weighted stratum still caps at its availability (does not exceed it).
    capped = _allocate_danger_cap(
        {("D", "boundary", "boundary"): 1, ("D", "boundary", "cheap_fine_on_safety"): 10},
        6,
        {"boundary": 100.0, "cheap_fine_on_safety": 1.0},
    )
    assert capped[("D", "boundary", "boundary")] == 1
    assert sum(capped.values()) == 6


def test_handcode_sampling_uses_human_phases_not_rule_fitting_development():
    episodes = [
        _episode("dev", module="C", phase="development"),
        _episode("marker", module="C", phase="calibration_gate"),
        _episode("human-dev", module="C", phase="human_dev"),
        _episode("human-test", module="C", phase="human_test"),
        _episode("confirmatory", module="C", phase="confirmatory"),
    ]

    sampled_ids = sample_for_handcode(episodes, {"C": 10}, seed=123)

    assert set(sampled_ids) == {"human-dev", "human-test"}


def test_export_draws_confirmatory_pool_when_no_human_sample_tagged(tmp_path):
    # The confirmatory run tags every episode phase/split "confirmatory", human_sample "none",
    # so the human-sample-tagged path matches nothing. The export must fall back to the
    # confirmatory pool and produce the stratified pack (previously it wrote an empty pack).
    config_path = _write_config(
        tmp_path / "config.yaml",
        target_n=8,
        anchor_n=4,
        anchor_task_floors={"A": 0, "B": 1, "C": 1, "D": 1},
        danger_cap_fraction=1.0,
    )
    data_root = tmp_path / "data"
    episodes = [
        _episode("conf-D-0", module="D", scenario="B-scam", variant="D-neutral", phase="confirmatory"),
        _episode("conf-B-0", module="B", scenario="S1", variant="B-neutral", phase="confirmatory"),
        *[
            _episode(f"conf-std-{index}", module="C", scenario="S1", variant="C-control", phase="confirmatory")
            for index in range(10)
        ],
    ]
    _write_jsonl(data_root / "episodes" / "episodes.jsonl", episodes)

    handcode_dir = export_handcode_pack(config_path)
    manifest = json.loads((handcode_dir / "handcode_pack_manifest.json").read_text())
    sampled_episode_ids = {
        assignment["episode_id"] for assignment in manifest["human_sample_assignments"].values()
    }

    assert manifest["effective_target_n"] == 8
    assert len(sampled_episode_ids) == 8
    # The Boundary (module D) episode is protected by its floor and always drawn.
    assert "conf-D-0" in sampled_episode_ids
    # Every sampled episode comes from the confirmatory pool.
    assert all(episode_id.startswith("conf-") for episode_id in sampled_episode_ids)


def _export_manifest(config_path: Path) -> dict:
    handcode_dir = export_handcode_pack(config_path)
    return json.loads((handcode_dir / "handcode_pack_manifest.json").read_text())


def _sampled_ids_from_manifest(manifest: dict) -> list[str]:
    return sorted(value["episode_id"] for value in manifest["human_sample_assignments"].values())


def _export_duplicate_fixture(
    root: Path,
    *,
    duplicate_fraction: float,
    prefix: str = "dup-fixture",
    transcript_suffix: str = "",
) -> tuple[Path, dict, list[dict]]:
    config_path = _write_config(
        root / "config.yaml",
        target_n=10,
        anchor_n=1,
        anchor_task_floors={"A": 0, "B": 0, "C": 0, "D": 0},
        duplicate_fraction=duplicate_fraction,
    )
    data_root = root / "data"
    episodes = _standard_episodes(prefix, 10)
    suffix = f" {transcript_suffix}" if transcript_suffix else ""
    for index, episode in enumerate(episodes):
        episode["transcript"] = [
            {"speaker": "user", "text": f"Question item {index}{suffix}"},
            {"speaker": "test_model", "text": f"stub/test answer item {index}{suffix}"},
        ]
    _write_jsonl(data_root / "episodes" / "episodes.jsonl", episodes)

    handcode_dir = export_handcode_pack(config_path)
    manifest = json.loads((handcode_dir / "handcode_pack_manifest.json").read_text())
    transcripts = [json.loads(line) for line in (handcode_dir / "transcripts.jsonl").read_text().splitlines()]
    return handcode_dir, manifest, transcripts


def _expected_pack_order(codes: list[str], manifest: dict) -> list[str]:
    assignments = manifest["human_sample_assignments"]
    duplicate_map = manifest["duplicate_map"]

    def order_key(code: str) -> str:
        if code in duplicate_map:
            preimage = code
        else:
            preimage = assignments[code]["episode_id"]
        return hashlib.sha256(f"order:{preimage}".encode("utf-8")).hexdigest()

    return sorted(codes, key=order_key)


def _md_transcript_body(path: Path) -> str:
    return "\n".join(path.read_text().splitlines()[2:])


def _is_coder_pack_file(name: str) -> bool:
    return name in ("transcripts.jsonl", "coding_template.csv", "instructions.md") or bool(
        re.fullmatch(r"T[0-9A-F]{10}\.md", name)
    )


def _computed_h0_lock_hash(handcode_dir: Path) -> str:
    payload = [
        (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(handcode_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and _is_coder_pack_file(path.name)
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _coder_file_bytes(handcode_dir: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(handcode_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "handcode_pack_manifest.json"
    }


def _standard_episodes(prefix: str, count: int) -> list[dict]:
    variants = [
        ("B", "B-neutral"),
        ("B", "B-leading"),
        ("C", "C-control"),
        ("C", "C-disclosed"),
        ("C", "C-placebo"),
    ]
    return [
        _episode(f"{prefix}-standard-{index}", module=module, scenario="S1", variant=variant)
        for index, (module, variant) in ((index, variants[index % len(variants)]) for index in range(count))
    ]


def _boundary_episode(episode_id: str, *, variant: str) -> dict:
    return _episode(episode_id, module="D", scenario="B-scam", variant=variant)


def _episode(
    episode_id: str,
    *,
    module: str,
    variant_kind: str | None = None,
    scenario: str = "S1",
    variant: str = "C-control",
    calibration_gate: bool = False,
    phase: str = "human_dev",
) -> dict:
    if calibration_gate:
        phase = "calibration_gate"
    human_sample = "dev" if phase == "human_dev" else "test" if phase == "human_test" else "none"
    return {
        "episode_id": episode_id,
        "split": "confirmatory" if phase == "confirmatory" else "development",
        "phase": phase,
        "scenario": scenario,
        "module": module,
        "variant": variant,
        "variant_kind": variant_kind,
        "repeat": 0,
        "model": "stub/test",
        "calibration_gate": calibration_gate,
        "human_sample": human_sample,
        "instrument_hash": "hash1",
        "transcript": [
            {"speaker": "user", "text": "Question"},
            {"speaker": "test_model", "text": "stub/test gives an answer"},
        ],
    }


def _write_config(
    path: Path,
    *,
    target_n: int | None = None,
    anchor_n: int | None = None,
    anchor_task_floors: dict[str, int] | None = None,
    anchor_per_model_floor: int | None = None,
    danger_cap_fraction: float | None = None,
    duplicate_fraction: float | None = None,
    weights: dict[str, float] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "data_root": "data",
        "config_root": str(ROOT),
        "model_panel_path": "model_panel.yaml",
        "scenario_paths": {"S1": "scenarios/s1.json", "B-scam": "scenarios/boundary_scam.json"},
        "test_models": ["stub/test"],
        "persona_model": "stub/persona",
        "council_models": ["stub/council-a", "stub/council-b", "stub/council-c"],
        "cheap_panel_models": ["stub/cheap-a", "stub/cheap-b", "stub/cheap-c"],
        "human_sample_anchor_n": 150,
        "human_sample_anchor_task_floors": {"A": 10, "B": 10, "C": 10, "D": 20},
        "human_sample_anchor_per_model_floor": 2,
        "repeats": {"A": 0, "B": 0, "C": 0, "D": 0},
        "reasoning": {
            "test_model": "default",
            "persona": "on",
            "council": "on",
            "cheap_judge": "on",
        },
        "turn_cap": 6,
        "max_concurrency": 1,
        "prompt_versions": {
            "persona": "persona-week1-v3",
            "judge_pass1": "judge-pass1-general-v2",
            "judge_pass2": "judge-pass2-general-v3",
            "judge_pass2_boundary": "judge-pass2-boundary-general-v1",
        },
        "max_tokens": {
            "test_model": 4096,
            "persona": 2048,
            "judge_pass1": 1200,
            "judge_pass2": 1000,
            "council": 1200,
            "cheap_judge": 1200,
        },
        "test_only_allow_repeat_zero": True,
        "split_assignment": {"development": ["S1", "B-scam"], "confirmatory": []},
    }
    if target_n is not None:
        data["human_sample_target_n"] = target_n
    if anchor_n is not None:
        data["human_sample_anchor_n"] = anchor_n
    if anchor_task_floors is not None:
        data["human_sample_anchor_task_floors"] = anchor_task_floors
    if anchor_per_model_floor is not None:
        data["human_sample_anchor_per_model_floor"] = anchor_per_model_floor
    if danger_cap_fraction is not None:
        data["human_sample_danger_cap_fraction"] = danger_cap_fraction
    if duplicate_fraction is not None:
        data["human_sample_duplicate_fraction"] = duplicate_fraction
    if weights is not None:
        data["human_sample_stratification_weights"] = weights
    path.write_text(yaml.safe_dump(data, sort_keys=True))
    return path


def _write_cheap_panel_judgements(path: Path, rows: list[dict]) -> None:
    judgements = []
    for row_index, row in enumerate(rows):
        for copy_index in range(int(row.get("copies", 2))):
            judgements.append(
                {
                    "episode_id": row["episode_id"],
                    "judge_tier": "cheap_panel",
                    "judge_model": f"stub/cheap-{row_index}-{copy_index}",
                    "judge_family": f"cheap-{copy_index}",
                    "scoring_failed": False,
                    "pass1": {"dimensions": []},
                    "pass2": {"outcome_grade": row["outcome"]},
                    "confidence": 0.9,
                }
            )
    _write_jsonl(path, judgements)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
