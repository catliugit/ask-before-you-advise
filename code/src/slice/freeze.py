from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Iterable

import yaml

from .gate import GATE_VERDICTS
from .schema import FROZEN_HASH_INPUTS, SliceConfig, load_config, load_prompt_file, resolve_from_config

FREEZE_SCRIPT_VERSION = "1"
FREEZE_RECORD_JSON = "freeze_record.json"
FREEZE_RECORD_MD = "freeze_record.md"
CALIBRATION_FROZEN_JSON = "calibration_frozen.json"
# Read-only back-compat for records created by the early hyphenated shim.
LEGACY_FREEZE_RECORD_JSON = "freeze-record.json"
FROZEN_GLOB_EXCLUDES = {
    "code/src/slice/*.py": frozenset({"code/src/slice/cli.py"}),
}
ENCODING_NOTE = (
    "Files are read as bytes, CRLF and lone CR are normalised to LF, then SHA-256 is "
    "computed. No JSON canonicalisation, whitespace stripping, case folding, or canary "
    "normalisation is applied."
)
ANALYSIS_BOUNDARY_NOTE = (
    "The freeze hashes config.yaml, the scenario bank, grading prompts and docs, the "
    "model panel, pre-registration, the analysis package, and the src/slice protocol "
    "package that turns prompts and saved conversations into grades and reported "
    "numbers. It excludes only cli.py, the command entry; freeze.py is included so "
    "hash lookup and validation protocol drift changes the frozen hash."
)
ACCEPTED_HASH_WINDOW_NOTE = (
    "Verification accepts only the original frozen instrument_hash and, when deviations "
    "exist, the latest logged deviation hash. Earlier deviation hashes do not verify."
)
PREFREEZE_PREREQUISITE_NOTE = (
    "Before the real freeze run, confirm the grading codebook prose is consistent "
    "with the settled reading of every pilot grade."
)
PROMPT_VERSION_FILES = {
    "persona": "code/prompts/persona.md",
    "judge_pass1": "code/prompts/judge_pass1.md",
    "judge_pass2": "code/prompts/judge_pass2.md",
    "judge_pass2_boundary": "code/prompts/judge_pass2_boundary.md",
    "judge_resist_initial": "code/prompts/judge_resist_initial.md",
    "council_coder": "code/prompts/council_coder.md",
    "prosecutor_pass": "code/prompts/prosecutor_pass.md",
}
CALIBRATION_VERDICTS = frozenset({"trusted", "escalate_whole_scenario"})
BOUNDARY_SAFETY_VERDICTS = frozenset({"PASS", "BLOCKED_FALSE_CLEAR", "BELOW_N_FLOOR", "NOT_BINDING"})


class FreezeError(RuntimeError):
    pass


class MissingFrozenInputError(FreezeError):
    pass


class FrozenRunHashError(FreezeError):
    pass


@dataclass(frozen=True)
class FreezeFileEntry:
    path: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    status: str
    record_path: Path
    expected_hash: str | None
    actual_hash: str
    accepted_hashes: tuple[str, ...]
    drifted_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "record_path": str(self.record_path),
            "expected_hash": self.expected_hash,
            "actual_hash": self.actual_hash,
            "accepted_hashes": list(self.accepted_hashes),
            "drifted_files": list(self.drifted_files),
        }


def gather_files(config_or_path: SliceConfig | str | Path) -> list[Path]:
    config = _ensure_config(config_or_path)
    repo_root = repo_root_for_config(config)
    paths: list[Path] = []
    missing: list[str] = []
    for declared in FROZEN_HASH_INPUTS:
        pattern = _declared_path(repo_root, declared)
        if _has_glob_meta(declared):
            matches = sorted(glob(str(pattern)), key=lambda value: _relative_sort_key(Path(value), repo_root))
            excluded = FROZEN_GLOB_EXCLUDES.get(declared, frozenset())
            matches = [
                match
                for match in matches
                if _relative_path(Path(match), repo_root) not in excluded
            ]
            if not matches:
                missing.append(declared)
                continue
            paths.extend(Path(match).resolve() for match in matches)
            continue
        path = pattern.resolve()
        if not path.is_file():
            missing.append(declared)
            continue
        paths.append(path)
    if missing:
        raise MissingFrozenInputError("missing frozen input(s): " + ", ".join(missing))
    return _sort_unique_paths(paths, repo_root)


def _normalise_bytes(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _distill_calibration_permission(record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "verdict",
        "audit_n_apparent_pass",
        "audit_n_non_pass",
        "false_safe_errors",
        "routine_disagree_pct",
        "human_items_audited",
        "council_items_audited",
    ]
    distilled = {
        key: record.get(key)
        for key in keys
        if key in record and _is_decision_scalar(record.get(key))
    }
    distilled["cheap_grading_permitted"] = True
    return distilled


def _distill_module_verdicts(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    distilled: dict[str, dict[str, Any]] = {}
    for module, record in sorted(block.items()):
        if not isinstance(record, dict):
            continue
        entry = {
            key: value
            for key, value in sorted(record.items())
            if key == "verdict" or _is_decision_scalar(value)
        }
        if entry:
            distilled[str(module)] = entry
    return distilled


def _distill_false_clear(gate_verdict: dict[str, Any]) -> dict[str, Any]:
    source = gate_verdict.get("false_clear", {}) if isinstance(gate_verdict.get("false_clear"), dict) else {}
    boundary = source.get("boundary", {}) if isinstance(source.get("boundary"), dict) else {}
    distilled_boundary = {
        key: value
        for key, value in sorted(boundary.items())
        if key in {"binding", "below_n_floor", "false_clear_count", "n_human_dangerous", "n_compared"}
        and _is_decision_scalar(value)
    }
    return {
        "boundary": distilled_boundary,
        "boundary_safety_verdict": gate_verdict.get(
            "boundary_safety_verdict",
            source.get("boundary_safety_verdict", "NOT_BINDING"),
        ),
        "safety_set_widened_required": bool(
            gate_verdict.get("safety_set_widened_required", source.get("safety_set_widened_required", False))
        ),
    }


def _is_decision_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _permission_audit_counts_all_zero(record: dict[str, Any]) -> bool:
    count_keys = ("audit_n_apparent_pass", "audit_n_non_pass", "human_items_audited", "council_items_audited")
    return all(not record.get(key) for key in count_keys)


def _frozen_calibration_active(config: SliceConfig) -> bool:
    panel_path = resolve_from_config(config, config.model_panel_path, root="config")
    try:
        panel = yaml.safe_load(panel_path.read_text())
    except OSError:
        panel = {}
    if isinstance(panel, dict) and panel.get("freeze_day"):
        return True
    return (repo_root_for_config(config) / FREEZE_RECORD_JSON).exists()


def _preflight_frozen_inputs(config: SliceConfig) -> list[dict[str, Any]]:
    repo_root = repo_root_for_config(config)
    violations = []
    for declared in FROZEN_HASH_INPUTS:
        pattern = _declared_path(repo_root, declared)
        if _has_glob_meta(declared):
            matches = sorted(glob(str(pattern)), key=lambda value: _relative_sort_key(Path(value), repo_root))
            excluded = FROZEN_GLOB_EXCLUDES.get(declared, frozenset())
            matches = [
                match
                for match in matches
                if _relative_path(Path(match), repo_root) not in excluded
            ]
            if not matches:
                violations.append(_violation("missing_frozen_input", f"missing frozen input(s): {declared}"))
            continue
        if not pattern.is_file():
            violations.append(_violation("missing_frozen_input", f"missing frozen input(s): {declared}"))
    return violations


def _preflight_model_panel(config: SliceConfig) -> list[dict[str, Any]]:
    path = resolve_from_config(config, config.model_panel_path, root="config")
    panel = yaml.safe_load(path.read_text())
    violations = []
    if not isinstance(panel, dict) or not panel.get("freeze_day"):
        violations.append(_violation("model_panel_freeze_day", "model_panel.yaml freeze_day must be non-null"))
    for entry in panel.get("entries", []) if isinstance(panel, dict) else []:
        if not entry.get("pinned_version"):
            violations.append(
                _violation(
                    "model_panel_pinned_version",
                    f"model_panel entry {entry.get('slug', '<unknown>')} pinned_version must be non-null",
                )
            )
    return violations


def _preflight_cost_ceiling(config: SliceConfig) -> list[dict[str, Any]]:
    ceiling = config.cost_ceiling
    if not isinstance(ceiling, (int, float)) or isinstance(ceiling, bool) or ceiling <= 0:
        return [_violation("cost_ceiling", "config.cost_ceiling must be a positive number")]
    return []


def _preflight_judge_cost_ceiling(config: SliceConfig) -> list[dict[str, Any]]:
    ceiling = config.judge_cost_ceiling
    if not isinstance(ceiling, (int, float)) or isinstance(ceiling, bool) or ceiling <= 0:
        return [_violation("judge_cost_ceiling", "config.judge_cost_ceiling must be a positive number")]
    return []


def _preflight_prompt_versions(config: SliceConfig) -> list[dict[str, Any]]:
    repo_root = repo_root_for_config(config)
    violations = []
    for key, expected in sorted(config.prompt_versions.items()):
        declared = PROMPT_VERSION_FILES.get(key)
        if declared is None:
            violations.append(_violation("prompt_version_unknown", f"no prompt file mapping for config.prompt_versions.{key}"))
            continue
        path = repo_root / declared
        try:
            actual = load_prompt_file(path).version
        except Exception as exc:  # noqa: BLE001 - preflight records and continues.
            violations.append(_violation("prompt_version_parse", f"{declared} could not be parsed: {exc}"))
            continue
        if actual != expected:
            violations.append(
                _violation(
                    "prompt_version_mismatch",
                    f"{declared} header version {actual!r} != config.prompt_versions.{key} {expected!r}",
                )
            )
    return violations


def _preflight_red_team_fixture(config: SliceConfig) -> list[dict[str, Any]]:
    if not config.red_team_fixture_path:
        return [_violation("red_team_fixture", "config.red_team_fixture_path must be set")]
    path = resolve_from_config(config, config.red_team_fixture_path, root="config")
    harmful = safe = 0
    violations = []
    try:
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            expected = item.get("expected")
            if expected == "harmful":
                harmful += 1
            elif expected == "safe":
                safe += 1
            else:
                violations.append(_violation("red_team_fixture_parse", f"{path}:{lineno} has invalid expected={expected!r}"))
    except Exception as exc:  # noqa: BLE001 - preflight records and continues.
        return [_violation("red_team_fixture_parse", f"red-team fixture does not parse: {exc}")]
    total = harmful + safe
    if harmful <= 0:
        violations.append(_violation("red_team_fixture_harmful_n", "red-team fixture harmful count must be > 0"))
    if safe <= 0:
        violations.append(_violation("red_team_fixture_safe_n", "red-team fixture safe count must be > 0"))
    if config.red_team_n != total:
        violations.append(
            _violation(
                "red_team_n_mismatch",
                f"config.red_team_n={config.red_team_n} but fixture total item count is {total} (harmful={harmful}, safe={safe})",
                {"n_probes": harmful, "n_safe": safe},
            )
        )
    return violations


def _preflight_calibration_distillate(config: SliceConfig) -> list[dict[str, Any]]:
    path = repo_root_for_config(config) / "code" / CALIBRATION_FROZEN_JSON
    try:
        distillate = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - preflight records and continues.
        return [_violation("calibration_frozen_parse", f"{path} does not parse: {exc}")]
    return validate_calibration_distillate(distillate, config)


def validate_calibration_distillate(distillate: dict[str, Any], config: SliceConfig) -> list[dict[str, Any]]:
    violations = []
    if not isinstance(distillate, dict):
        return [_violation("calibration_frozen_schema", "calibration_frozen.json must be a JSON object")]
    permissions = distillate.get("calibration_permissions")
    gate = distillate.get("gate")
    if not isinstance(permissions, dict):
        violations.append(_violation("calibration_frozen_schema", "calibration_frozen.json missing calibration_permissions object"))
        permissions = {}
    if not isinstance(gate, dict):
        violations.append(_violation("calibration_frozen_schema", "calibration_frozen.json missing gate object"))
        gate = {}
    for scenario_id in config.effective_phase_assignment.confirmatory:
        if scenario_id not in permissions:
            violations.append(
                _violation("calibration_frozen_permission", f"missing frozen calibration permission for confirmatory scenario {scenario_id}")
            )
    for scenario_id, record in permissions.items():
        if not isinstance(record, dict) or "cheap_grading_permitted" not in record or record.get("verdict") not in CALIBRATION_VERDICTS:
            violations.append(_violation("calibration_frozen_permission", f"invalid frozen calibration permission for scenario {scenario_id}"))
    # Stub guard: a real distillate always carries at least one non-zero audit count somewhere
    # (the calibration gate counted every episode). All-zero counts across every scenario is the
    # signature of a placeholder file, which must never be frozen.
    permission_records = [record for record in permissions.values() if isinstance(record, dict)]
    if permission_records and all(_permission_audit_counts_all_zero(record) for record in permission_records):
        violations.append(
            _violation(
                "calibration_frozen_stub",
                "calibration_frozen.json carries all-zero audit counts in every calibration permission; "
                "it looks like a placeholder stub - regenerate it from the live calibration outputs "
                "(slice.cli freeze-calibration)",
            )
        )
    for block_name in ("per_module", "council_vs_human"):
        block = gate.get(block_name)
        if not isinstance(block, dict):
            violations.append(_violation("calibration_frozen_gate", f"calibration_frozen gate missing required block {block_name}"))
            continue
        for module in ("A", "B", "C", "D"):
            verdict = block.get(module, {}).get("verdict") if isinstance(block.get(module), dict) else None
            if verdict not in GATE_VERDICTS:
                violations.append(_violation("calibration_frozen_gate", f"invalid {block_name}.{module}.verdict {verdict!r}"))
    false_clear = gate.get("false_clear")
    if not isinstance(false_clear, dict):
        violations.append(_violation("calibration_frozen_gate", "calibration_frozen gate missing required block false_clear"))
    elif false_clear.get("boundary_safety_verdict") not in BOUNDARY_SAFETY_VERDICTS:
        violations.append(
            _violation(
                "calibration_frozen_gate",
                f"invalid false_clear.boundary_safety_verdict {false_clear.get('boundary_safety_verdict')!r}",
            )
        )
    return violations


def _violation(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"code": code, "message": message}
    if details:
        out["details"] = details
    return out


def hash_file(path: str | Path) -> str:
    return hashlib.sha256(_normalised_file_bytes(Path(path))).hexdigest()


def compute_instrument_hash(config_or_path: SliceConfig | str | Path) -> tuple[str, list[FreezeFileEntry]]:
    config = _ensure_config(config_or_path)
    repo_root = repo_root_for_config(config)
    entries: list[FreezeFileEntry] = []
    manifest_lines: list[str] = []
    for path in gather_files(config):
        rel_path = _relative_path(path, repo_root)
        data = _normalised_file_bytes(path)
        digest = hashlib.sha256(data).hexdigest()
        entries.append(FreezeFileEntry(path=rel_path, sha256=digest, bytes=len(data)))
        manifest_lines.append(f"{digest}  {rel_path}\n")
    top_digest = hashlib.sha256("".join(manifest_lines).encode("utf-8")).hexdigest()
    return top_digest, entries


def freeze_calibration(
    config_or_path: SliceConfig | str | Path,
    *,
    gate_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    config = _ensure_config(config_or_path)
    data_root = Path(config.data_root)
    gate_source = Path(gate_path) if gate_path is not None else data_root / "outputs" / "gate_verdict.json"
    calibration_source = (
        Path(calibration_path)
        if calibration_path is not None
        else data_root / "outputs" / "calibration_verdicts.json"
    )
    if not gate_source.exists():
        raise FreezeError(f"missing gate verdict sidecar: {gate_source}")
    if not calibration_source.exists():
        raise FreezeError(f"missing calibration verdict sidecar: {calibration_source}")
    distillate = build_calibration_distillate(
        json.loads(gate_source.read_text()),
        json.loads(calibration_source.read_text()),
    )
    output = Path(output_path) if output_path is not None else repo_root_for_config(config) / "code" / CALIBRATION_FROZEN_JSON
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(distillate, indent=2, sort_keys=True) + "\n")
    return output


def build_calibration_distillate(
    gate_verdict: dict[str, Any],
    calibration_verdicts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "calibration_permissions": {
            str(scenario_id): _distill_calibration_permission(record)
            for scenario_id, record in sorted(calibration_verdicts.items())
        },
        "gate": {
            "per_module": _distill_module_verdicts(gate_verdict.get("per_module", {})),
            "council_vs_human": _distill_module_verdicts(gate_verdict.get("council_vs_human", {})),
            "false_clear": _distill_false_clear(gate_verdict),
        },
    }


def load_calibration_distillate(config_or_path: SliceConfig | str | Path) -> dict[str, Any] | None:
    config = _ensure_config(config_or_path)
    path = repo_root_for_config(config) / "code" / CALIBRATION_FROZEN_JSON
    if not path.exists() or not _frozen_calibration_active(config):
        return None
    return json.loads(path.read_text())


def calibration_verdicts_from_distillate(distillate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    permissions = distillate.get("calibration_permissions", {})
    return {
        str(scenario_id): {
            "scenario_id": str(scenario_id),
            "verdict": record.get("verdict"),
            "false_safe_errors": record.get("false_safe_errors", 0),
            "routine_disagree_pct": record.get("routine_disagree_pct", 0.0),
            "audit_n_apparent_pass": record.get("audit_n_apparent_pass", 0),
            "audit_n_non_pass": record.get("audit_n_non_pass", 0),
            "human_items_audited": record.get("human_items_audited", 0),
            "council_items_audited": record.get("council_items_audited", 0),
        }
        for scenario_id, record in permissions.items()
        if record.get("cheap_grading_permitted") is True
    }


def gate_verdict_from_distillate(distillate: dict[str, Any]) -> dict[str, Any]:
    gate = dict(distillate.get("gate", {}))
    false_clear = dict(gate.get("false_clear", {}))
    if false_clear:
        gate["false_clear"] = false_clear
        gate["boundary_safety_verdict"] = false_clear.get("boundary_safety_verdict", "NOT_BINDING")
        gate["safety_set_widened_required"] = bool(false_clear.get("safety_set_widened_required", False))
    return gate


def freeze_preflight(config_or_path: SliceConfig | str | Path) -> dict[str, Any]:
    config = _ensure_config(config_or_path)
    violations: list[dict[str, Any]] = []
    violations.extend(_preflight_frozen_inputs(config))
    violations.extend(_preflight_model_panel(config))
    violations.extend(_preflight_cost_ceiling(config))
    violations.extend(_preflight_judge_cost_ceiling(config))
    violations.extend(_preflight_prompt_versions(config))
    violations.extend(_preflight_red_team_fixture(config))
    violations.extend(_preflight_calibration_distillate(config))
    return {"ok": not violations, "violations": violations}


def assert_freeze_preflight(config_or_path: SliceConfig | str | Path) -> None:
    preflight = freeze_preflight(config_or_path)
    if not preflight["ok"]:
        messages = "; ".join(violation["message"] for violation in preflight["violations"])
        raise FreezeError("freeze preflight failed: " + messages)


def write_freeze_record(
    config_or_path: SliceConfig | str | Path,
    *,
    date_stamp: str,
    external_timestamp: dict[str, Any] | None = None,
    record_dir: str | Path | None = None,
    enforce_preflight: bool = True,
) -> dict[str, Any]:
    if not date_stamp or not str(date_stamp).strip():
        raise ValueError("date_stamp is required")
    config = _ensure_config(config_or_path)
    if enforce_preflight:
        assert_freeze_preflight(config)
    repo_root = repo_root_for_config(config)
    out_dir = Path(record_dir).resolve() if record_dir is not None else repo_root
    out_dir.mkdir(parents=True, exist_ok=True)

    instrument_hash, entries = compute_instrument_hash(config)
    model_panel_path = resolve_from_config(config, config.model_panel_path, root="config")
    model_panel_snapshot = yaml.safe_load(model_panel_path.read_text())
    groups = _group_entries(entries)
    git_commit, git_clean = _git_state(repo_root)
    record = {
        "instrument_hash": instrument_hash,
        "date_stamp": date_stamp,
        "freeze_timestamp": date_stamp,
        "model_panel_freeze_day": model_panel_snapshot.get("freeze_day") if isinstance(model_panel_snapshot, dict) else None,
        "tool_versions": _tool_versions(repo_root),
        "encoding_note": ENCODING_NOTE,
        "analysis_boundary_note": ANALYSIS_BOUNDARY_NOTE,
        "accepted_hash_window_note": ACCEPTED_HASH_WINDOW_NOTE,
        "pre_freeze_prerequisite_note": PREFREEZE_PREREQUISITE_NOTE,
        "git_commit": git_commit,
        "git_clean": git_clean,
        "external_timestamp": external_timestamp or _default_external_timestamp(git_commit),
        "model_panel_snapshot": model_panel_snapshot,
        "files": [entry.to_dict() for entry in entries],
        "config_files": groups["config_files"],
        "scenario_files": groups["scenario_files"],
        "prompt_files": groups["prompt_files"],
        "analysis_files": groups["analysis_files"],
        "instrument_docs": groups["instrument_docs"],
        "model_panel": groups["model_panel"],
        "pre_registration": groups["pre_registration"],
        "deviations": [],
    }
    _write_record_files(out_dir, record)
    return record


def verify(
    config_or_path: SliceConfig | str | Path,
    *,
    record_path: str | Path | None = None,
    enforce_preflight: bool = True,
) -> VerifyResult:
    config = _ensure_config(config_or_path)
    if enforce_preflight:
        assert_freeze_preflight(config)
    record, path = _load_freeze_record(config, record_path=record_path)
    actual_hash, entries = compute_instrument_hash(config)
    accepted_hashes = _accepted_hashes(record)
    if actual_hash in accepted_hashes:
        status = "OK" if actual_hash == record.get("instrument_hash") else "OK_LOGGED_DEVIATION"
        return VerifyResult(
            ok=True,
            status=status,
            record_path=path,
            expected_hash=record.get("instrument_hash"),
            actual_hash=actual_hash,
            accepted_hashes=accepted_hashes,
            drifted_files=(),
        )
    drifted = _drifted_files(record.get("files", []), entries)
    return VerifyResult(
        ok=False,
        status="MISMATCH",
        record_path=path,
        expected_hash=record.get("instrument_hash"),
        actual_hash=actual_hash,
        accepted_hashes=accepted_hashes,
        drifted_files=tuple(drifted),
    )


def load_frozen_hash(config_or_path: SliceConfig | str | Path) -> str | None:
    try:
        record, _ = _load_freeze_record(_ensure_config(config_or_path))
    except FileNotFoundError:
        return None
    value = record.get("instrument_hash")
    return str(value) if isinstance(value, str) and value else None


def stamp_instrument_hash(
    record: dict[str, Any],
    config_or_path: SliceConfig | str | Path,
    *,
    enforce_preflight: bool = True,
) -> dict[str, Any]:
    stamped = dict(record)
    split = stamped.get("split")
    if split == "confirmatory":
        if enforce_preflight:
            assert_freeze_preflight(config_or_path)
        frozen_hash = load_frozen_hash(config_or_path)
        if frozen_hash is None:
            raise FrozenRunHashError("confirmatory record cannot be stamped because no freeze_record.json exists")
        stamped["instrument_hash"] = frozen_hash
    elif split == "development":
        stamped["instrument_hash"] = None
    return stamped


def assert_frozen_confirmatory_episodes(
    episodes: Iterable[dict[str, Any]],
    expected_hash: str,
) -> None:
    bad: list[str] = []
    for episode in episodes:
        if episode.get("split") != "confirmatory":
            continue
        if episode.get("instrument_hash") != expected_hash:
            bad.append(str(episode.get("episode_id", "<unknown>")))
    if bad:
        raise FrozenRunHashError(
            "confirmatory episode instrument_hash missing or mismatched: " + ", ".join(sorted(bad))
        )


def assert_frozen_run_episodes(
    config_or_path: SliceConfig | str | Path,
    episodes: Iterable[dict[str, Any]],
    *,
    enforce_preflight: bool = True,
) -> None:
    if enforce_preflight:
        assert_freeze_preflight(config_or_path)
    frozen_hash = load_frozen_hash(config_or_path)
    if frozen_hash is None:
        raise FrozenRunHashError("frozen run requires freeze_record.json before confirmatory episodes are accepted")
    assert_frozen_confirmatory_episodes(episodes, frozen_hash)


def append_deviation(
    config_or_path: SliceConfig | str | Path,
    *,
    file: str,
    what_changed: str,
    why: str,
    date_stamp: str,
    record_path: str | Path | None = None,
) -> str:
    if not date_stamp or not str(date_stamp).strip():
        raise ValueError("date_stamp is required")
    config = _ensure_config(config_or_path)
    repo_root = repo_root_for_config(config)
    record, path = _load_freeze_record(config, record_path=record_path)
    new_hash, entries = compute_instrument_hash(config)
    named_file = _normalise_record_file(file, repo_root)
    frozen_paths = {entry.path for entry in entries}
    if named_file not in frozen_paths:
        raise FreezeError(f"deviation file is not a frozen input: {file}")
    drifted = _drifted_files(record.get("files", []), entries)
    if named_file not in drifted:
        raise FreezeError(f"deviation file has not drifted: {named_file}")
    deviations = list(record.get("deviations", []))
    deviations.append(
        {
            "date": date_stamp,
            "file": named_file,
            "what_changed": what_changed,
            "why": why,
            "drifted_files": drifted,
            "new_instrument_hash": new_hash,
        }
    )
    record["deviations"] = deviations
    _refresh_record_files(record, entries)
    _write_record_files(path.parent, record)
    return new_hash


def repo_root_for_config(config: SliceConfig) -> Path:
    config_root = Path(config.config_root).resolve()
    if (config_root / "scenarios").is_dir() and config_root.name == "code":
        return config_root.parent
    if (config_root / "code").is_dir():
        return config_root
    return config_root.parent


def run_cli(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog="python -m slice.freeze")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--date-stamp")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--freeze-calibration", action="store_true")
    parser.add_argument("--freeze-preflight", action="store_true")
    parser.add_argument("--add-deviation", nargs=3, metavar=("FILE", "WHAT_CHANGED", "WHY"))
    parser.add_argument("--external-timestamp-method")
    parser.add_argument("--external-timestamp-reference")
    parser.add_argument("--external-timestamp-instructions")
    args = parser.parse_args(argv)

    if args.verify:
        result = verify(args.config)
        return result.to_dict()
    if args.freeze_calibration:
        path = freeze_calibration(args.config)
        return {"calibration_frozen": str(path)}
    if args.freeze_preflight:
        return freeze_preflight(args.config)
    if args.add_deviation:
        if not args.date_stamp:
            raise ValueError("--date-stamp is required with --add-deviation")
        new_hash = append_deviation(
            args.config,
            file=args.add_deviation[0],
            what_changed=args.add_deviation[1],
            why=args.add_deviation[2],
            date_stamp=args.date_stamp,
        )
        return {"new_instrument_hash": new_hash}
    if not args.date_stamp:
        raise ValueError("--date-stamp is required when writing freeze_record.json")
    record = write_freeze_record(
        args.config,
        date_stamp=args.date_stamp,
        external_timestamp=_cli_external_timestamp(args),
    )
    return {"instrument_hash": record["instrument_hash"], "record": FREEZE_RECORD_JSON}


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run_cli(argv), sort_keys=True))


def _ensure_config(config_or_path: SliceConfig | str | Path) -> SliceConfig:
    if isinstance(config_or_path, SliceConfig):
        return config_or_path
    return load_config(config_or_path)


def _declared_path(repo_root: Path, declared: str) -> Path:
    path = Path(declared)
    if path.is_absolute():
        return path
    return repo_root / path


def _has_glob_meta(value: str) -> bool:
    return any(char in value for char in "*?[")


def _sort_unique_paths(paths: list[Path], repo_root: Path) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in sorted(paths, key=lambda item: _relative_sort_key(item, repo_root)):
        rel = _relative_path(path, repo_root)
        if rel in seen:
            raise FreezeError(f"duplicate frozen input after glob expansion: {rel}")
        seen.add(rel)
        result.append(path)
    return result


def _relative_sort_key(path: Path, repo_root: Path) -> bytes:
    return _relative_path(path.resolve(), repo_root).encode("utf-8")


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _normalised_file_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise MissingFrozenInputError(f"missing frozen input: {path}")
    return _normalise_bytes(path.read_bytes())


def _group_entries(entries: list[FreezeFileEntry]) -> dict[str, list[str]]:
    paths = [entry.path for entry in entries]
    return {
        "config_files": [path for path in paths if path == "code/config.yaml"],
        "scenario_files": [path for path in paths if path.startswith("code/scenarios/") and path.endswith(".json")],
        "prompt_files": [path for path in paths if path.startswith("code/prompts/")],
        "analysis_files": [
            path
            for path in paths
            if (path.startswith("code/src/slice/") and path.endswith(".py")) or path == "code/compute_kappa.py"
        ],
        "instrument_docs": [
            path
            for path in paths
            if path in {"grading-codebook.md", "decision-rules.md", "severity-rubric.md"}
        ],
        "model_panel": [path for path in paths if path == "code/model_panel.yaml"],
        "pre_registration": [path for path in paths if path == "pre-registration.md"],
    }


def _refresh_record_files(record: dict[str, Any], entries: list[FreezeFileEntry]) -> None:
    groups = _group_entries(entries)
    record["files"] = [entry.to_dict() for entry in entries]
    record["config_files"] = groups["config_files"]
    record["scenario_files"] = groups["scenario_files"]
    record["prompt_files"] = groups["prompt_files"]
    record["analysis_files"] = groups["analysis_files"]
    record["instrument_docs"] = groups["instrument_docs"]
    record["model_panel"] = groups["model_panel"]
    record["pre_registration"] = groups["pre_registration"]


def _normalise_record_file(value: str, repo_root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return _relative_path(path, repo_root)
    return path.as_posix().lstrip("./")


def _git_state(repo_root: Path) -> tuple[str | None, bool]:
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    status = _run_git(repo_root, "status", "--porcelain")
    if commit is None:
        return None, False
    return commit, status == ""


def _tool_versions(repo_root: Path) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "git": _run_git(repo_root, "--version"),
        "freeze_script_version": FREEZE_SCRIPT_VERSION,
    }


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _default_external_timestamp(git_commit: str | None) -> dict[str, str]:
    return {
        "method": "local-git-commit-only",
        "reference": git_commit or "no-git-commit",
        "verified_instructions": (
            "This is a local pointer only. On the real freeze day, add a GitHub commit/tag, "
            "OSF registration, or OpenTimestamps proof and record that external reference here."
        ),
    }


def _write_record_files(out_dir: Path, record: dict[str, Any]) -> None:
    (out_dir / FREEZE_RECORD_JSON).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    (out_dir / FREEZE_RECORD_MD).write_text(_render_markdown(record))


def _render_markdown(record: dict[str, Any]) -> str:
    short_hash = str(record["instrument_hash"])[:12]
    lines = [
        f"# Freeze record {short_hash}",
        "",
        f"- Instrument hash: `{record['instrument_hash']}`",
        f"- Date stamp: `{record['date_stamp']}`",
        f"- File count: {len(record.get('files', []))}",
        f"- External timestamp: `{record.get('external_timestamp', {}).get('method')}` "
        f"{record.get('external_timestamp', {}).get('reference')}",
        "",
        "## Notes",
        "",
        ANALYSIS_BOUNDARY_NOTE,
        "",
        ACCEPTED_HASH_WINDOW_NOTE,
        "",
        PREFREEZE_PREREQUISITE_NOTE,
        "",
        "## Files",
        "",
        "| Path | SHA-256 | Bytes |",
        "|---|---:|---:|",
    ]
    for entry in record.get("files", []):
        lines.append(f"| `{entry['path']}` | `{entry['sha256'][:12]}` | {entry['bytes']} |")
    lines.extend(
        [
            "",
            "## Deviations",
            "",
            "| Date | File | Drifted files | Change | Why | New hash |",
            "|---|---|---|---|---|---|",
        ]
    )
    deviations = record.get("deviations", [])
    if deviations:
        for deviation in deviations:
            drifted_files = ", ".join(f"`{path}`" for path in deviation.get("drifted_files", [deviation["file"]]))
            lines.append(
                f"| {deviation['date']} | `{deviation['file']}` | {drifted_files} | {deviation['what_changed']} | "
                f"{deviation['why']} | `{deviation['new_instrument_hash'][:12]}` |"
            )
    else:
        lines.append("| none |  |  |  |  |  |")
    return "\n".join(lines) + "\n"


def _load_freeze_record(
    config: SliceConfig,
    *,
    record_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    if record_path is not None:
        path = Path(record_path)
        if not path.is_absolute():
            path = repo_root_for_config(config) / path
        return json.loads(path.read_text()), path.resolve()
    repo_root = repo_root_for_config(config)
    search_dirs = _freeze_record_search_dirs(config, repo_root)
    for directory in search_dirs:
        for name in (FREEZE_RECORD_JSON, LEGACY_FREEZE_RECORD_JSON):
            path = directory / name
            if path.exists():
                return json.loads(path.read_text()), path
    raise FileNotFoundError(repo_root / FREEZE_RECORD_JSON)


def _freeze_record_search_dirs(config: SliceConfig, repo_root: Path) -> list[Path]:
    candidates = [Path(config.data_root).resolve().parent, repo_root.resolve()]
    seen: set[Path] = set()
    result = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def _accepted_hashes(record: dict[str, Any]) -> tuple[str, ...]:
    hashes = [str(record["instrument_hash"])]
    deviations = record.get("deviations", [])
    if deviations:
        latest = deviations[-1].get("new_instrument_hash")
        if latest:
            hashes.append(str(latest))
    return tuple(hashes)


def _drifted_files(record_files: list[dict[str, Any]], current_entries: list[FreezeFileEntry]) -> list[str]:
    recorded = {str(entry["path"]): str(entry["sha256"]) for entry in record_files}
    current = {entry.path: entry.sha256 for entry in current_entries}
    paths = sorted(set(recorded) | set(current), key=lambda value: value.encode("utf-8"))
    drifted = []
    for path in paths:
        if recorded.get(path) != current.get(path):
            drifted.append(path)
    return drifted


def _cli_external_timestamp(args: argparse.Namespace) -> dict[str, str] | None:
    if not (args.external_timestamp_method or args.external_timestamp_reference or args.external_timestamp_instructions):
        return None
    if not args.external_timestamp_method or not args.external_timestamp_reference:
        raise ValueError("--external-timestamp-method and --external-timestamp-reference must be supplied together")
    return {
        "method": args.external_timestamp_method,
        "reference": args.external_timestamp_reference,
        "verified_instructions": args.external_timestamp_instructions or "operator-provided external timestamp pointer",
    }


if __name__ == "__main__":
    main()
