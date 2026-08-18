from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ._jsonl import append_jsonl
from .client import OpenRouterClient
from .judge import (
    _blind_transcript_for_judge,
    _cheap_jobs_for_episode,
    _chat_json,
    _cited_rule_for_track,
    _council_jobs_for_episode,
    _judge_family_map,
    _load_all_scenarios,
    _max_tokens,
    _prompt_dir,
    _read_jsonl,
    judge_episode,
)
from .kappa_gate import _consensus_labels_by_episode, _false_clear_rate_ucb, _safety_binary
from .schema import (
    RedTeamProbeItem,
    RedTeamProbeRecord,
    Scenario,
    load_config,
    load_prompt_file,
    model_prices_for_config,
    model_to_dict,
    resolve_from_config,
)


logger = logging.getLogger(__name__)


def run_red_team_probe(
    *,
    n_harmful: int,
    n_slipped: int,
    n_safe: int,
    n_over_flagged: int,
    n_prosecutor_harmful: int,
    n_prosecutor_harmful_tripped: int,
    n_prosecutor_safe: int,
    n_prosecutor_safe_tripped: int,
    run_timestamp: str,
    max_pass_rate: float,
    tripwire_min_fire: float,
    tripwire_max_false_fire: float,
    slip_route_breakdown: dict[str, int] | None = None,
    n_graders_skipped: int = 0,
) -> RedTeamProbeRecord:
    """Pure red-team probe summary from already tallied counts."""

    _check_count("n_harmful", n_harmful)
    _check_count("n_slipped", n_slipped, denominator=n_harmful)
    _check_count("n_safe", n_safe)
    _check_count("n_over_flagged", n_over_flagged, denominator=n_safe)
    _check_count("n_prosecutor_harmful", n_prosecutor_harmful)
    _check_count(
        "n_prosecutor_harmful_tripped",
        n_prosecutor_harmful_tripped,
        denominator=n_prosecutor_harmful,
    )
    _check_count("n_prosecutor_safe", n_prosecutor_safe)
    _check_count("n_prosecutor_safe_tripped", n_prosecutor_safe_tripped, denominator=n_prosecutor_safe)

    cheap_pass_rate = n_slipped / n_harmful if n_harmful else 0.0
    over_flag_rate = n_over_flagged / n_safe if n_safe else 0.0
    tripwire_fire_rate_harmful = (
        n_prosecutor_harmful_tripped / n_prosecutor_harmful if n_prosecutor_harmful else 0.0
    )
    tripwire_fire_rate_safe = n_prosecutor_safe_tripped / n_prosecutor_safe if n_prosecutor_safe else 0.0
    tripwire_discriminates = (
        None
        if n_prosecutor_harmful == 0 or n_prosecutor_safe == 0
        else tripwire_fire_rate_harmful >= tripwire_min_fire
        and tripwire_fire_rate_safe <= tripwire_max_false_fire
    )
    safety_set_expanded = cheap_pass_rate > max_pass_rate

    return RedTeamProbeRecord(
        run_timestamp=run_timestamp,
        n_probes=n_harmful,
        n_cleared_by_cheap=n_slipped,
        cheap_pass_rate=cheap_pass_rate,
        cheap_pass_rate_ucb=_false_clear_rate_ucb(n_slipped, n_harmful),
        n_safe_probes=n_safe,
        n_over_flagged=n_over_flagged,
        over_flag_rate=over_flag_rate,
        over_flag_rate_ucb=_false_clear_rate_ucb(n_over_flagged, n_safe),
        n_prosecutor_harmful=n_prosecutor_harmful,
        tripwire_fire_rate_harmful=tripwire_fire_rate_harmful,
        n_prosecutor_safe=n_prosecutor_safe,
        tripwire_fire_rate_safe=tripwire_fire_rate_safe,
        tripwire_discriminates=tripwire_discriminates,
        safety_set_expanded=safety_set_expanded,
        n_graders_skipped=n_graders_skipped,
        expansion_reason=(
            _expansion_reason(cheap_pass_rate, max_pass_rate, n_slipped, slip_route_breakdown)
            if safety_set_expanded
            else None
        ),
    )


def run_red_team_probe_from_config(config_path: str | Path, client: Any | None = None) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    fixture_path = _fixture_path(config)
    prosecutor_model = config.effective_prosecutor_model
    if prosecutor_model is None:
        raise ValueError("red-team probe requires effective_prosecutor_model; set prosecutor_model or council_models")

    if client is None:
        client = OpenRouterClient(
            cache_dir=resolve_from_config(config_path, config.cache_dir),
            cost_log_path=Path(config.data_root) / "cost_log.jsonl",
            bad_output_retries=int(config.retries.get("bad_output", 3)),
            model_prices=model_prices_for_config(config),
            retry_max_tokens_cap=config.retry_max_tokens_cap,
        )

    scenarios = _load_all_scenarios(config)
    family_map = _judge_family_map(config)
    prompt_dir = _prompt_dir(config_path, config)
    pass1_prompt = load_prompt_file(prompt_dir / "judge_pass1.md")
    pass2_prompt = load_prompt_file(prompt_dir / "judge_pass2.md")
    pass2_boundary_prompt = load_prompt_file(prompt_dir / "judge_pass2_boundary.md")
    resist_initial_prompt = load_prompt_file(prompt_dir / "judge_resist_initial.md")
    prosecutor_prompt = load_prompt_file(prompt_dir / "prosecutor_pass.md")

    n_harmful = 0
    n_slipped = 0
    n_safe = 0
    n_over_flagged = 0
    n_prosecutor_harmful = 0
    n_prosecutor_harmful_tripped = 0
    n_prosecutor_safe = 0
    n_prosecutor_safe_tripped = 0
    n_graders_skipped = 0
    slip_route_breakdown: dict[str, int] = {}
    probe_judgements_path = Path(config.data_root) / "probe_judgements.jsonl"
    persist_probe_judgements = bool(getattr(config, "probe_judgements_enabled", False))
    # Resume guard: probe rows are keyed (probe_id, judge_tier, judge_model), mirroring
    # score_all's judgement skip key. A re-invocation (smoke -> full run on the shared
    # data_root, or a resume after a ceiling stop) must never append duplicate rows —
    # _council_probe_miss_summary counts votes by appending rows per probe, so duplicates
    # would silently distort the council vote. Same stale-grade caveat as judgements.jsonl:
    # a prompt bump requires quarantining the old rows (the key carries no prompt version).
    existing_probe_rows: set[tuple[str, str, str]] = set()
    if persist_probe_judgements and probe_judgements_path.exists():
        existing_probe_rows = {
            (str(row.get("probe_id")), str(row.get("judge_tier")), str(row.get("judge_model")))
            for row in _read_jsonl(probe_judgements_path)
        }

    def _persist_probe_row(judgement: dict[str, Any]) -> None:
        key = (str(judgement.get("probe_id")), str(judgement.get("judge_tier")), str(judgement.get("judge_model")))
        if key in existing_probe_rows:
            return
        existing_probe_rows.add(key)
        append_jsonl(probe_judgements_path, judgement)

    items = list(_load_fixture_items(fixture_path))
    for item_index, item in enumerate(items):
        # The main judging batches are ceiling-guarded in _run_judgement_jobs; the probe loop
        # spends through the same client and must respect the same ceiling. Checked per item
        # (not mid-item) so every processed probe has its complete set of grader rows; the
        # resume guard above makes a stopped probe run cleanly resumable.
        judge_ceiling = getattr(config, "judge_cost_ceiling", None)
        if judge_ceiling is not None and getattr(client, "session_cost", 0.0) >= judge_ceiling:
            logger.warning(
                "Judge cost ceiling %s reached at %.4f during red-team probe; %d of %d probe items skipped (resumable via red-team).",
                judge_ceiling,
                getattr(client, "session_cost", 0.0),
                len(items) - item_index,
                len(items),
            )
            break
        scenario = _scenario_for_item(item, scenarios)
        _validate_item_reference(item, scenario)
        episode = _episode_from_item(item, run_timestamp=config.run_timestamp, prompt_versions=config.prompt_versions)

        cheap_rows = []
        for job in _cheap_jobs_for_episode(config, episode, family_map, calibration_verdicts=None):
            try:
                judgement = _tag_probe_judgement(
                    judge_episode(
                        config,
                        job,
                        scenario,
                        pass1_prompt,
                        pass2_prompt,
                        client,
                        pass2_boundary_prompt=pass2_boundary_prompt,
                        resist_initial_prompt=resist_initial_prompt,
                    ),
                    item,
                )
                if persist_probe_judgements:
                    _persist_probe_row(judgement)
                cheap_rows.append(judgement)
            except Exception:
                # A single cheap grader exhausting its bad-output retries must NOT abort the whole
                # probe (mirrors score_all's per-episode resilience). Skip this grader; the danger
                # judgement is taken over the graders that succeeded. NOTE the consensus is a majority
                # over PRESENT labels (kappa_gate._consensus_labels_by_episode needs >=2 agreeing
                # labels), so dropping a grader from a 2-1 majority can flip an item from caught to
                # slipped (or an over-flag from counted to not). The bias is CONSERVATIVE (it
                # over-reports slips / under-reports over-flags, the safe direction), and every skip
                # is surfaced in n_graders_skipped so a flaky-grader-driven count is interpretable.
                n_graders_skipped += 1
                logger.exception(
                    "Red-team cheap grade failed for item=%s model=%s; skipping this grader",
                    item.item_id,
                    job.judge_model,
                )
        if not cheap_rows:
            raise ValueError(
                f"red-team item {item.item_id}: every cheap grader failed; cannot score it"
            )
        if persist_probe_judgements:
            for job in _council_jobs_for_episode(config, episode, family_map, require_rule_fitting=False):
                try:
                    _persist_probe_row(
                        _tag_probe_judgement(
                            judge_episode(
                                config,
                                job,
                                scenario,
                                pass1_prompt,
                                pass2_prompt,
                                client,
                                pass2_boundary_prompt=pass2_boundary_prompt,
                                resist_initial_prompt=resist_initial_prompt,
                            ),
                            item,
                        ),
                    )
                except Exception:
                    logger.exception(
                        "Red-team council/shadow grade failed for item=%s model=%s; skipping this grader",
                        item.item_id,
                        job.judge_model,
                    )
        caught = _danger_caught_by_cheap_consensus(
            episode,
            cheap_rows,
            deferral_dangerous_max=config.safety_dangerous_deferral_max,
        )
        tripwire = _prosecutor_tripwire(
            config,
            client,
            episode,
            scenario,
            prosecutor_prompt,
            prosecutor_model,
        )

        if item.expected == "harmful":
            n_harmful += 1
            n_prosecutor_harmful += 1
            if not caught:
                n_slipped += 1
                key = f"module_{item.module}"
                slip_route_breakdown[key] = slip_route_breakdown.get(key, 0) + 1
            if tripwire:
                n_prosecutor_harmful_tripped += 1
        else:
            n_safe += 1
            n_prosecutor_safe += 1
            if caught:
                n_over_flagged += 1
            if tripwire:
                n_prosecutor_safe_tripped += 1

    record = run_red_team_probe(
        n_harmful=n_harmful,
        n_slipped=n_slipped,
        n_safe=n_safe,
        n_over_flagged=n_over_flagged,
        n_prosecutor_harmful=n_prosecutor_harmful,
        n_prosecutor_harmful_tripped=n_prosecutor_harmful_tripped,
        n_prosecutor_safe=n_prosecutor_safe,
        n_prosecutor_safe_tripped=n_prosecutor_safe_tripped,
        run_timestamp=config.run_timestamp,
        max_pass_rate=config.red_team_max_pass_rate,
        tripwire_min_fire=config.red_team_tripwire_min_fire,
        tripwire_max_false_fire=config.red_team_tripwire_max_false_fire,
        slip_route_breakdown=slip_route_breakdown,
        n_graders_skipped=n_graders_skipped,
    )
    output = Path(config.data_root) / "outputs" / "red_team_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model_to_dict(record), indent=2, sort_keys=True) + "\n")
    return output


def _fixture_path(config: Any) -> Path:
    if config.red_team_fixture_path is None:
        raise ValueError("red_team_fixture_path is required for the red-team probe")
    path = resolve_from_config(config, config.red_team_fixture_path, root="config")
    if not path.exists():
        raise FileNotFoundError(f"red_team_fixture_path does not exist: {path}")
    return path


def _tag_probe_judgement(judgement: dict[str, Any], item: RedTeamProbeItem) -> dict[str, Any]:
    tagged = dict(judgement)
    tagged["is_probe"] = True
    tagged["probe_expected"] = item.expected
    tagged["probe_id"] = item.item_id
    tagged["stratum"] = item.stratum or f"{item.scenario}:module_{item.module}:{item.variant}:{item.expected}"
    tagged["scenario"] = item.scenario
    tagged["module"] = item.module
    tagged["variant"] = item.variant
    return tagged


def _load_fixture_items(path: Path) -> list[RedTeamProbeItem]:
    items: list[RedTeamProbeItem] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            try:
                item = (
                    RedTeamProbeItem.model_validate(data)
                    if hasattr(RedTeamProbeItem, "model_validate")
                    else RedTeamProbeItem.parse_obj(data)
                )
            except Exception as exc:
                raise ValueError(f"malformed red-team fixture item at {path}:{line_number}: {exc}") from exc
            items.append(item)
    return items


def _scenario_for_item(item: RedTeamProbeItem, scenarios: dict[str, Scenario]) -> Scenario:
    try:
        return scenarios[item.scenario]
    except KeyError as exc:
        raise ValueError(f"red-team item {item.item_id} references unknown scenario {item.scenario}") from exc


def _validate_item_reference(item: RedTeamProbeItem, scenario: Scenario) -> None:
    module = getattr(scenario, f"module_{item.module.lower()}", None)
    if module is None:
        raise ValueError(f"red-team item {item.item_id} references missing module {item.module} on {scenario.id}")
    if not any(variant.id == item.variant for variant in module.variants):
        raise ValueError(
            f"red-team item {item.item_id} references unknown variant "
            f"{scenario.id}/{item.module}/{item.variant}"
        )


def _episode_from_item(
    item: RedTeamProbeItem,
    *,
    run_timestamp: str,
    prompt_versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "episode_id": item.item_id,
        "run_id": "red-team-probe",
        "split": "development",
        "phase": "development",
        "run_timestamp": run_timestamp,
        "model": "red-team-fixture",
        "observed_model_version": None,
        "scenario": item.scenario,
        "module": item.module,
        "variant": item.variant,
        "repeat": 0,
        "prompt_versions": dict(prompt_versions),
        "transcript": item.transcript,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
        "cost": 0.0,
        "effective_temperature": None,
        "reasoning_setting": "default",
        "instrument_hash": None,
        "persona_leak": False,
        "canary_leaks": [],
        "rerun_due_to_persona_leak": False,
        "rerun_count": 0,
        "call_status": "ok",
        "retry_count": 0,
        "calibration_gate": False,
        "human_sample": "none",
        "establishment": [],
        "capitulation": None,
        "failure_reason": None,
    }


def _danger_caught_by_cheap_consensus(
    episode: dict[str, Any],
    cheap_rows: list[dict[str, Any]],
    *,
    deferral_dangerous_max: int,
) -> bool:
    consensus = _consensus_labels_by_episode(cheap_rows, tier="cheap_panel")
    labels = consensus.get(str(episode["episode_id"]), {}).get("labels", {})
    return (
        _safety_binary(episode, labels, deferral_dangerous_max=deferral_dangerous_max)
        == "dangerous"
    )


def _prosecutor_tripwire(
    config: Any,
    client: Any,
    episode: dict[str, Any],
    scenario: Scenario,
    prosecutor_prompt: Any,
    model: str,
) -> bool:
    try:
        transcript = _blind_transcript_for_judge(episode, scenario)
        payload = {
            "scenario_id": scenario.id,
            "module": episode["module"],
            "regulatory_track": scenario.regulatory_track,
            "cited_rule": _cited_rule_for_track(scenario),
            "signpost_context": scenario.signpost_context,
            "transcript": transcript,
        }
        _result, verdict = _chat_json(
            config,
            client,
            episode,
            "prosecutor",
            model,
            [
                {"role": "system", "content": prosecutor_prompt.text},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0,
            max_tokens=_max_tokens(config, "prosecutor", 600),
            timestamp=config.run_timestamp,
            expected_json_keys={"argument": str, "tripwire": bool},
            reasoning_key="council",
        )
        return bool(verdict["tripwire"])
    except Exception:
        logger.exception("Red-team prosecutor failed for episode_id=%s", episode.get("episode_id"))
        return True


def _check_count(name: str, value: int, *, denominator: int | None = None) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if denominator is not None and value > denominator:
        raise ValueError(f"{name} cannot exceed its denominator")


def _expansion_reason(
    cheap_pass_rate: float,
    max_pass_rate: float,
    n_slipped: int,
    slip_route_breakdown: dict[str, int] | None,
) -> str:
    breakdown = slip_route_breakdown or {"unclassified": n_slipped}
    formatted = ",".join(f"{key}:{breakdown[key]}" for key in sorted(breakdown))
    return (
        f"cheap_pass_rate={cheap_pass_rate:.6g} exceeds max_pass_rate={max_pass_rate:.6g}; "
        f"slip_breakdown={formatted}"
    )
