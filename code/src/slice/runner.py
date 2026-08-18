from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import inspect
import json
import logging
from pathlib import Path
import threading
from typing import Any
import uuid

from ._jsonl import append_jsonl, replace_jsonl_by_key
from .canary import detect_leak, elicited, present_in_prompt
from .client import ChatResult, OpenRouterClient
from .freeze import FrozenRunHashError, load_frozen_hash, verify
from .persona import FORCING_MOVE, build_persona_messages, build_persona_system_prompt, unasked_facts
from .schema import (
    Dimension,
    Episode,
    ModuleAVariant,
    PromptVariant,
    Scenario,
    SliceConfig,
    load_config,
    load_scenario,
    model_prices_for_config,
    model_to_dict,
    resolve_reasoning,
    resolve_from_config,
    validate_instrument,
)


logger = logging.getLogger(__name__)

EpisodeJob = tuple[Scenario, str, Any, str, int, str]
PHASES = ("development", "calibration_gate", "human_dev", "human_test", "confirmatory")
PROMPT_VERSION_FILES = {
    "persona": "prompts/persona.md",
    "judge_pass1": "prompts/judge_pass1.md",
    "judge_pass2": "prompts/judge_pass2.md",
    "judge_pass2_boundary": "prompts/judge_pass2_boundary.md",
    "judge_resist_initial": "prompts/judge_resist_initial.md",
    "council_coder": "prompts/council_coder.md",
    "prosecutor_pass": "prompts/prosecutor_pass.md",
}


def run_all(
    config_path: str | Path,
    client: Any | None = None,
    *,
    retry_missing: bool = False,
    enforce_preflight: bool = True,
) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    data_dir = Path(config.data_root)
    episodes_path = data_dir / "episodes" / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)

    if client is None:
        client = OpenRouterClient(
            cache_dir=Path(config.cache_dir),
            cost_log_path=data_dir / "cost_log.jsonl",
            bad_output_retries=_bad_output_retries(config),
            model_prices=model_prices_for_config(config),
            retry_max_tokens_cap=config.retry_max_tokens_cap,
        )

    scenarios = _load_all_scenarios(config)
    validate_instrument(
        list(scenarios.values()),
        drop_false_premise=config.cut_stage.drop_false_premise,
        prompt_versions=config.prompt_versions,
        prompt_paths=_prompt_version_paths(config),
    )
    existing = _existing_episode_ids(episodes_path, retry_missing=retry_missing)
    known_missing = _missing_episode_ids(episodes_path) if retry_missing else set()
    jobs = _pending_episode_jobs(config, scenarios, existing)
    frozen_hash = _frozen_hash_for_run(config, jobs, enforce_preflight=enforce_preflight)
    run_id = _make_run_id()
    _write_run_manifest(config, run_id, data_dir)
    spend_tracker = _SpendTracker(config.cost_ceiling)
    persona_summary = {"total": 0, "rerun_once": 0, "double_leak": 0}

    with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
        pending: deque[EpisodeJob] = deque(jobs)
        active: dict[Future[tuple[str, float, dict[str, Any]]], EpisodeJob] = {}
        ceiling_reached = False

        def submit_next() -> None:
            if not pending or spend_tracker.ceiling_exceeded():
                return
            job = pending.popleft()
            future = executor.submit(
                _run_and_append_episode,
                config,
                config_path,
                frozen_hash,
                run_id,
                client,
                episodes_path,
                job,
                retry_missing,
                known_missing,
            )
            active[future] = job

        while pending and len(active) < config.max_concurrency and not spend_tracker.ceiling_exceeded():
            submit_next()

        if not active and pending and spend_tracker.ceiling_exceeded():
            _record_unsubmitted_missing(
                config,
                frozen_hash,
                run_id,
                episodes_path,
                list(pending),
                "cost_ceiling_reached",
                existing,
                replace_existing=retry_missing,
                known_missing=known_missing,
            )
            pending.clear()

        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    episode_id, cost, episode = future.result()
                except Exception as exc:
                    missing = _missing_episode_record(config, run_id, job, exc)
                    _stamp_episode_hash(missing, frozen_hash)
                    _write_episode_record(episodes_path, missing, replace_existing=retry_missing, known_missing=known_missing)
                    episode_id = missing["episode_id"]
                    cost = 0.0
                    episode = missing
                    logger.error("MISSING cell recorded: %s: %s", episode_id, exc)
                existing.add(episode_id)
                spend_tracker.add(cost)
                _update_persona_summary(persona_summary, episode)

                if spend_tracker.ceiling_exceeded() and not ceiling_reached:
                    ceiling_reached = True
                    unsubmitted = list(pending)
                    pending.clear()
                    _record_unsubmitted_missing(
                        config,
                        frozen_hash,
                        run_id,
                        episodes_path,
                        unsubmitted,
                        "cost_ceiling_reached",
                        existing,
                        replace_existing=retry_missing,
                        known_missing=known_missing,
                    )
                    logger.warning(
                        "Cost ceiling %s reached at %.4f. %d cells recorded as MISSING.",
                        config.cost_ceiling,
                        spend_tracker.total,
                        len(unsubmitted),
                    )

            while (
                pending
                and not ceiling_reached
                and len(active) < config.max_concurrency
                and not spend_tracker.ceiling_exceeded()
            ):
                submit_next()

    _log_persona_summary(persona_summary)
    return episodes_path


def _prompt_version_paths(config: SliceConfig) -> dict[str, Path]:
    return {
        key: resolve_from_config(config, relative_path, root="config")
        for key, relative_path in PROMPT_VERSION_FILES.items()
        if key in config.prompt_versions
    }


def _pending_episode_jobs(
    config: SliceConfig,
    scenarios: dict[str, Scenario],
    existing: set[str],
) -> list[EpisodeJob]:
    jobs: list[EpisodeJob] = []
    scheduled = set(existing)

    def add_job(scenario: Scenario, module: str, variant: Any, model: str, repeat: int, phase: str) -> None:
        episode_id = _episode_id(
            scenario.id,
            module,
            variant.id,
            model,
            repeat,
            phase=phase,
            human_sample=_human_sample_for_phase(phase),
        )
        if episode_id in scheduled:
            return
        scheduled.add(episode_id)
        jobs.append((scenario, module, variant, model, repeat, phase))

    for phase, scenario_id in _assigned_phase_scenario_ids(config):
        scenario = scenarios[scenario_id]
        for model in config.test_models:
            for module, variants in _scenario_variants(config, scenario):
                for variant in variants:
                    for repeat in range(config.repeats[module]):
                        add_job(scenario, module, variant, model, repeat, phase)
    return jobs


def _run_and_append_episode(
    config: SliceConfig,
    config_path: str | Path,
    frozen_hash: str | None,
    run_id: str,
    client: Any,
    episodes_path: Path,
    job: EpisodeJob,
    replace_existing: bool = False,
    known_missing: set[str] | None = None,
) -> tuple[str, float, dict[str, Any]]:
    scenario, module, variant, model, repeat, phase = job
    if module == "A":
        episode = run_module_a_episode(config, config_path, scenario, variant, model, repeat, client, run_id=run_id, phase=phase)
    else:
        episode = run_prompt_episode(config, scenario, module, variant, model, repeat, client, run_id=run_id, phase=phase)
    _stamp_episode_hash(episode, frozen_hash)
    _write_episode_record(episodes_path, episode, replace_existing=replace_existing, known_missing=known_missing)
    return episode["episode_id"], float(episode.get("cost", 0.0) or 0.0), episode


def _write_episode_record(
    path: Path,
    record: dict[str, Any],
    *,
    replace_existing: bool,
    known_missing: set[str] | None = None,
) -> None:
    # In a --retry-missing run replace_existing is True for every write, but only the rows
    # already in the file (the recorded MISSING cells) need the full-file rewrite that
    # replace_jsonl_by_key performs. Brand-new ids can be appended in O(1). When known_missing
    # is supplied we restrict the rewrite to those ids, so a recovery run is O(missing) rewrites
    # rather than O(n) per write. Passing None preserves replace-on-every-write. The decision
    # keys on the real record id, so a never-seen id is always appended (never duplicated).
    should_replace = replace_existing
    if should_replace and known_missing is not None:
        should_replace = record["episode_id"] in known_missing
    if should_replace:
        replace_jsonl_by_key(path, record, key="episode_id")
        return
    append_jsonl(path, record)


def _write_run_manifest(config: SliceConfig, run_id: str, data_dir: Path) -> None:
    outputs_dir = data_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "created_at": _wall_clock_timestamp(),
        "cut_stage": model_to_dict(config.cut_stage),
        "frame_capture": {
            "mode": config.cut_stage.fcr_mode,
            "demoted_to_exploratory": config.cut_stage.fcr_mode == "exploratory",
            "demotion_reasons": _fcr_demotion_reasons(config),
        },
    }
    (outputs_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def _fcr_demotion_reasons(config: SliceConfig) -> list[str]:
    reasons = []
    if config.cut_stage.drop_pushback:
        reasons.append("cut_stage.drop_pushback")
    if config.cut_stage.drop_repeats_to_2:
        reasons.append("cut_stage.drop_repeats_to_2")
    return reasons


def run_module_a_episode(
    config: SliceConfig,
    config_path: str | Path,
    scenario: Scenario,
    variant: ModuleAVariant,
    model: str,
    repeat: int,
    client: Any,
    *,
    run_id: str | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    run_id = run_id or _make_run_id()
    episode = _run_module_a_attempt(config, run_id, scenario, variant, model, repeat, client, phase=phase)
    if episode["persona_leak"]:
        rerun = _run_module_a_attempt(config, run_id, scenario, variant, model, repeat, client, phase=phase)
        rerun["rerun_due_to_persona_leak"] = True
        rerun["rerun_count"] = 2 if rerun["persona_leak"] else 1
        if rerun["persona_leak"]:
            logger.warning("Double persona leak for episode_id=%s. Accepted with contamination flag.", rerun["episode_id"])
        return _validate_episode(rerun)
    episode["rerun_count"] = 0
    episode["rerun_due_to_persona_leak"] = False
    return _validate_episode(episode)


def _run_module_a_attempt(
    config: SliceConfig,
    run_id: str,
    scenario: Scenario,
    variant: ModuleAVariant,
    model: str,
    repeat: int,
    client: Any,
    *,
    phase: str | None = None,
) -> dict[str, Any]:
    episode_id = _episode_id(
        scenario.id,
        "A",
        variant.id,
        model,
        repeat,
        phase=phase,
        human_sample=_human_sample_for_phase(phase),
    )
    transcript: list[dict[str, Any]] = []
    results: list[Any] = []
    elicited_dims: set[str] = set()
    leaks: list[dict[str, str]] = []
    persona_model: str | None = None
    persona_observed_model_version: str | None = None

    prompt_versions = dict(config.prompt_versions)
    persona_prompt_path = _prompt_dir(config) / "persona.md"
    persona_version, persona_system_prompt = build_persona_system_prompt(persona_prompt_path, variant)
    prompt_versions["persona"] = persona_version

    initial_prompt = variant.prompt if variant.kind == "fully_specified" else scenario.surface_prompt
    if initial_prompt is None:
        raise ValueError("Module A requires a prompt")
    transcript.append(_turn("user", "user", initial_prompt))
    messages = [{"role": "user", "content": initial_prompt}]

    model_turn_count = 0
    while model_turn_count < config.turn_cap:
        result = _chat(
            client,
            model,
            messages,
            temperature=None,
            max_tokens=_max_tokens(config, "test_model", 4096),
            cache_scope=episode_id,
            timestamp=config.run_timestamp,
            reasoning=_reasoning(config, "test_model", model),
            role="test_model",
            episode_id=episode_id,
        )
        results.append(result)
        model_text = _result_text(result)
        model_turn_count += 1
        transcript.append(
            _turn(
                "assistant",
                "test_model",
                model_text,
                truncated=_result_finish_reason(result) == "length",
            )
        )
        if _result_finish_reason(result) == "length":
            break

        new_question_dims = _newly_elicited_question_dims(model_text, scenario.dimensions, elicited_dims)
        if not new_question_dims:
            break
        elicited_dims.update(new_question_dims)

        messages.append({"role": "assistant", "content": model_text})
        if model_turn_count >= config.turn_cap:
            persona_text = FORCING_MOVE
            transcript.append(_turn("user", "persona", persona_text))
            messages.append({"role": "user", "content": persona_text})
            break

        persona_result = _chat(
            client,
            config.persona_model,
            build_persona_messages(persona_system_prompt, model_text),
            temperature=None,
            max_tokens=_max_tokens(config, "persona", 2048),
            cache_scope=episode_id,
            timestamp=config.run_timestamp,
            reasoning=_reasoning(config, "persona", config.persona_model),
            role="persona",
            episode_id=episode_id,
        )
        results.append(persona_result)
        persona_model = config.persona_model
        persona_observed_model_version = getattr(persona_result, "model_version", None)
        persona_text = _result_text(persona_result)
        new_leaks = detect_leak(persona_text, unasked_facts(variant, elicited_dims))
        leaks.extend(new_leaks)
        transcript.append(_turn("user", "persona", persona_text))
        messages.append({"role": "user", "content": persona_text})

    return _episode_record(
        config=config,
        run_id=run_id,
        scenario=scenario,
        module="A",
        variant_id=variant.id,
        model=model,
        repeat=repeat,
        prompt_versions=prompt_versions,
        variant=variant,
        opening_prompt=initial_prompt,
        transcript=transcript,
        results=results,
        persona_model=persona_model,
        persona_observed_model_version=persona_observed_model_version,
        persona_leak=bool(leaks),
        canary_leaks=leaks,
        rerun_due_to_persona_leak=False,
        rerun_count=0,
        capitulation=None,
        phase=phase,
    )


def run_prompt_episode(
    config: SliceConfig,
    scenario: Scenario,
    module: str,
    variant: PromptVariant,
    model: str,
    repeat: int,
    client: Any,
    *,
    run_id: str | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    run_id = run_id or _make_run_id()
    episode_id = _episode_id(
        scenario.id,
        module,
        variant.id,
        model,
        repeat,
        phase=phase,
        human_sample=_human_sample_for_phase(phase),
    )
    if variant.prompt is None:
        raise ValueError(f"Module {module} variant {variant.id} requires a prompt")
    transcript = [_turn("user", "user", variant.prompt)]
    messages = [{"role": "user", "content": variant.prompt}]
    results: list[Any] = []
    capitulation = None

    first = _chat(
        client,
        model,
        messages,
        temperature=None,
        max_tokens=_max_tokens(config, "test_model", 4096),
        cache_scope=episode_id,
        timestamp=config.run_timestamp,
        reasoning=_reasoning(config, "test_model", model),
        role="test_model",
        episode_id=episode_id,
    )
    results.append(first)
    first_text = _result_text(first)
    transcript.append(
        _turn(
            "assistant",
            "test_model",
            first_text,
            truncated=_result_finish_reason(first) == "length",
        )
    )
    if _result_finish_reason(first) == "length":
        return _validate_episode(
            _episode_record(
                config=config,
                run_id=run_id,
                scenario=scenario,
                module=module,
                variant_id=variant.id,
                model=model,
                repeat=repeat,
                prompt_versions=dict(config.prompt_versions),
                variant=variant,
                opening_prompt=variant.prompt,
                transcript=transcript,
                results=results,
                persona_leak=False,
                canary_leaks=[],
                rerun_due_to_persona_leak=False,
                rerun_count=0,
                capitulation=None,
                phase=phase,
            )
        )
    messages.append({"role": "assistant", "content": first_text})

    if (
        module == "B"
        and variant.variant_kind == "plain"
        and variant.pushback is not None
        and not config.cut_stage.drop_pushback
    ):
        transcript.append(_turn("user", "user", variant.pushback))
        messages.append({"role": "user", "content": variant.pushback})
        second = _chat(
            client,
            model,
            messages,
            temperature=None,
            max_tokens=_max_tokens(config, "test_model", 4096),
            cache_scope=episode_id,
            timestamp=config.run_timestamp,
            reasoning=_reasoning(config, "test_model", model),
            role="test_model",
            episode_id=episode_id,
        )
        results.append(second)
        transcript.append(
            _turn(
                "assistant",
                "test_model",
                _result_text(second),
                truncated=_result_finish_reason(second) == "length",
            )
        )
        capitulation = {
            "pushback_fired": True,
            "pre_pushback_text_index": 1,
            "post_pushback_text_index": len(transcript) - 1,
        }

    return _validate_episode(
        _episode_record(
            config=config,
            run_id=run_id,
            scenario=scenario,
            module=module,
            variant_id=variant.id,
            model=model,
            repeat=repeat,
            prompt_versions=dict(config.prompt_versions),
            variant=variant,
            opening_prompt=variant.prompt,
            transcript=transcript,
            results=results,
            persona_leak=False,
            canary_leaks=[],
            rerun_due_to_persona_leak=False,
            rerun_count=0,
            capitulation=capitulation,
            phase=phase,
        )
    )


def _episode_record(
    *,
    config: SliceConfig,
    run_id: str,
    scenario: Scenario,
    module: str,
    variant_id: str,
    model: str,
    repeat: int,
    prompt_versions: dict[str, str],
    variant: Any,
    opening_prompt: str,
    transcript: list[dict[str, Any]],
    results: list[Any],
    persona_leak: bool,
    canary_leaks: list[dict[str, str]],
    rerun_due_to_persona_leak: bool,
    rerun_count: int,
    capitulation: dict[str, Any] | None,
    phase: str | None,
    persona_model: str | None = None,
    persona_observed_model_version: str | None = None,
) -> dict[str, Any]:
    usage = _sum_usage(results)
    test_result = _result_for_model(results, model)
    phase_value = phase or _phase_for(config, scenario.id)
    human_sample = _human_sample_for_phase(phase_value)
    return {
        "episode_id": _episode_id(
            scenario.id,
            module,
            variant_id,
            model,
            repeat,
            phase=phase,
            human_sample=human_sample if phase is not None else None,
        ),
        "run_id": run_id,
        "split": _split_for_phase(phase_value),
        "phase": phase_value,
        "run_timestamp": _wall_clock_timestamp(),
        "model": model,
        "observed_model_version": getattr(test_result, "model_version", None),
        "persona_model": persona_model,
        "persona_observed_model_version": persona_observed_model_version,
        "scenario": scenario.id,
        "module": module,
        "variant": variant_id,
        "repeat": repeat,
        "prompt_versions": prompt_versions,
        "transcript": transcript,
        "usage": usage,
        "cost": usage["cost"],
        "effective_temperature": getattr(test_result, "sent_temperature", None),
        "reasoning_setting": _sent_reasoning_setting(test_result),
        "instrument_hash": None,
        "persona_leak": persona_leak,
        "canary_leaks": canary_leaks,
        "rerun_due_to_persona_leak": rerun_due_to_persona_leak,
        "rerun_count": rerun_count,
        "call_status": "ok",
        "retry_count": _retry_count(results),
        "calibration_gate": _calibration_gate_for_phase(phase_value),
        "human_sample": human_sample,
        "establishment": _establishment(scenario, opening_prompt, variant),
        "capitulation": capitulation,
        "failure_reason": None,
    }


def _sum_usage(results: list[Any]) -> dict[str, float]:
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens: int | None = None
    cost = 0.0
    for result in results:
        usage = getattr(result, "usage", {}) or {}
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        if usage.get("reasoning_tokens") is not None:
            reasoning_tokens = (reasoning_tokens or 0) + int(usage.get("reasoning_tokens", 0) or 0)
        cost += float(getattr(result, "cost_estimate", 0.0) or 0.0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost": cost,
    }


def _validate_episode(record: dict[str, Any]) -> dict[str, Any]:
    return model_to_dict(Episode.model_validate(record))


def _result_for_model(results: list[Any], model: str) -> Any | None:
    for result in results:
        result_model = getattr(result, "sent_model", None) or getattr(result, "model", None)
        if result_model == model:
            return result
    return None


def _sent_reasoning_setting(result: Any | None) -> str:
    if result is None:
        return "default"
    sent = getattr(result, "sent_reasoning", None)
    if sent in {"default", "on", "off", "minimal", "low", "medium", "high", "xhigh"}:
        return str(sent)
    return "default"


def _retry_count(results: list[Any]) -> int:
    # Cross-role total by decision: sums bad-output retries across every call made for the
    # episode, including the persona turns folded in for Module A, to match how _sum_usage
    # aggregates cost and tokens across roles. retry_count is a persisted diagnostic only and
    # is not read by the confirmatory analysis, so a cross-role total is the consistent choice.
    return sum(int(getattr(result, "retry_count", 0) or 0) for result in results)


def _establishment(scenario: Scenario, opening_prompt: str, variant: Any | None = None) -> list[dict[str, Any]]:
    facts_by_dimension: dict[str, list[Any]] = {}
    for fact in getattr(variant, "facts", []) or []:
        facts_by_dimension.setdefault(fact.dimension_id, []).append(fact)
    return [
        {
            "dimension_id": dimension.id,
            "present_in_prompt": present_in_prompt(opening_prompt, dimension)
            or any(present_in_prompt(opening_prompt, fact) for fact in facts_by_dimension.get(dimension.id, [])),
            "asked_for": False,
            "branch_covered": False,
        }
        for dimension in scenario.dimensions
    ]


def _missing_episode_record(
    config: SliceConfig,
    run_id: str,
    job: EpisodeJob,
    exc: Exception | str,
) -> dict[str, Any]:
    scenario, module, variant, model, repeat, phase = job
    reason = str(exc)
    human_sample = _human_sample_for_phase(phase)
    record = {
        "episode_id": _episode_id(
            scenario.id,
            module,
            variant.id,
            model,
            repeat,
            phase=phase,
            human_sample=human_sample,
        ),
        "run_id": run_id,
        "split": _split_for_phase(phase),
        "phase": phase,
        "run_timestamp": _wall_clock_timestamp(),
        "model": model,
        "observed_model_version": None,
        "persona_model": None,
        "persona_observed_model_version": None,
        "scenario": scenario.id,
        "module": module,
        "variant": variant.id,
        "repeat": repeat,
        "prompt_versions": dict(config.prompt_versions),
        "transcript": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": None, "cost": 0.0},
        "cost": 0.0,
        "effective_temperature": None,
        "reasoning_setting": str(config.reasoning.get("test_model", "default")),
        "instrument_hash": None,
        "persona_leak": False,
        "canary_leaks": [],
        "rerun_due_to_persona_leak": False,
        "rerun_count": 0,
        "call_status": "missing",
        "retry_count": 0,
        "calibration_gate": _calibration_gate_for_phase(phase),
        "human_sample": human_sample,
        "establishment": _establishment(
            scenario,
            getattr(variant, "prompt", None) or scenario.surface_prompt or "",
            variant,
        ),
        "capitulation": None,
        "failure_reason": reason,
    }
    return _validate_episode(record)


def _stamp_episode_hash(record: dict[str, Any], frozen_hash: str | None) -> None:
    if _is_confirmatory_record(record) and frozen_hash:
        record["instrument_hash"] = frozen_hash
    elif record.get("split") == "development":
        record["instrument_hash"] = None


def _record_unsubmitted_missing(
    config: SliceConfig,
    frozen_hash: str | None,
    run_id: str,
    episodes_path: Path,
    jobs: list[EpisodeJob],
    reason: str,
    existing: set[str],
    *,
    replace_existing: bool = False,
    known_missing: set[str] | None = None,
) -> None:
    for job in jobs:
        missing = _missing_episode_record(config, run_id, job, reason)
        _stamp_episode_hash(missing, frozen_hash)
        _write_episode_record(episodes_path, missing, replace_existing=replace_existing, known_missing=known_missing)
        existing.add(missing["episode_id"])


def _result_text(result: Any) -> str:
    if isinstance(result, ChatResult):
        return result.text.strip()
    return str(getattr(result, "text", "")).strip()


def _result_finish_reason(result: Any) -> str | None:
    finish_reason = getattr(result, "finish_reason", None)
    return str(finish_reason) if finish_reason is not None else None


def _newly_elicited_question_dims(
    model_text: str,
    dimensions: list[Dimension],
    already_elicited: set[str],
) -> set[str]:
    return {
        dimension.id
        for dimension in dimensions
        if dimension.id not in already_elicited and elicited(model_text, dimension)
    }


def _max_tokens(config: SliceConfig, key: str, default: int) -> int:
    return int(config.max_tokens.get(key, default))


def _reasoning(config: SliceConfig, key: str, model: str | None = None) -> str:
    return resolve_reasoning(config, key, model)


def _bad_output_retries(config: SliceConfig) -> int:
    return int(config.retries.get("bad_output", 3))


def _load_all_scenarios(config: SliceConfig) -> dict[str, Scenario]:
    scenarios: dict[str, Scenario] = {}
    for scenario_id in _assigned_scenario_ids(config):
        try:
            scenario_path = config.scenario_paths[scenario_id]
        except KeyError as exc:
            raise ValueError(f"split_assignment references unknown scenario {scenario_id!r}") from exc
        scenario = load_scenario(resolve_from_config(config, scenario_path, root="config"))
        scenarios[scenario.id] = scenario
    return scenarios


def _assigned_scenario_ids(config: SliceConfig) -> list[str]:
    ids = [scenario_id for _, scenario_id in _assigned_phase_scenario_ids(config)]
    seen: set[str] = set()
    ordered = []
    for scenario_id in ids:
        if scenario_id in seen:
            continue
        seen.add(scenario_id)
        ordered.append(scenario_id)
    return ordered


def _assigned_phase_scenario_ids(config: SliceConfig) -> list[tuple[str, str]]:
    assignment = config.effective_phase_assignment
    ordered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for phase in PHASES:
        for scenario_id in getattr(assignment, phase):
            key = (phase, scenario_id)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    return ordered


def _scenario_variants(config: SliceConfig, scenario: Scenario) -> list[tuple[str, list[Any]]]:
    modules: list[tuple[str, list[Any]]] = []
    if scenario.module_a is not None:
        modules.append(("A", _scheduled_variants(config, scenario.module_a.variants)))
    if scenario.module_b is not None:
        modules.append(("B", _scheduled_variants(config, scenario.module_b.variants)))
    if scenario.module_c is not None:
        modules.append(("C", _scheduled_variants(config, scenario.module_c.variants)))
    if scenario.module_d is not None:
        modules.append(("D", _scheduled_variants(config, scenario.module_d.variants)))
    return modules


def _scheduled_variants(config: SliceConfig, variants: list[Any]) -> list[Any]:
    scheduled = []
    for variant in variants:
        if config.cut_stage.drop_false_premise and getattr(variant, "variant_kind", None) == "false_premise":
            continue
        if (
            config.cut_stage.drop_second_boundary_wording
            and getattr(variant, "variant_kind", None) == "boundary"
            and _is_second_boundary_wording(variant)
        ):
            continue
        scheduled.append(variant)
    return scheduled


def _is_second_boundary_wording(variant: Any) -> bool:
    if getattr(variant, "is_second_wording", None) is True:
        return True
    wording_rank = getattr(variant, "wording_rank", None)
    return wording_rank is not None and int(wording_rank) > 1


def _split_for(config: SliceConfig, scenario_id: str) -> str:
    return _split_for_phase(_phase_for(config, scenario_id))


def _phase_for(config: SliceConfig, scenario_id: str) -> str:
    for phase in PHASES:
        if scenario_id in getattr(config.effective_phase_assignment, phase):
            return phase
    raise ValueError(f"scenario {scenario_id!r} is not present in split_assignment")


def _split_for_phase(phase: str) -> str:
    return "confirmatory" if phase == "confirmatory" else "development"


def _calibration_gate_for_phase(phase: str | None) -> bool:
    return phase == "calibration_gate"


def _human_sample_for_phase(phase: str | None) -> str:
    if phase == "human_dev":
        return "dev"
    if phase == "human_test":
        return "test"
    return "none"


def _is_confirmatory_record(record: dict[str, Any]) -> bool:
    return record.get("phase") == "confirmatory" or record.get("split") == "confirmatory"


def _frozen_hash_for_run(
    config: SliceConfig,
    jobs: list[EpisodeJob],
    *,
    enforce_preflight: bool = True,
) -> str | None:
    if not any(job[5] == "confirmatory" for job in jobs):
        return load_frozen_hash(config)
    try:
        verification = verify(config, enforce_preflight=enforce_preflight)
    except FileNotFoundError as exc:
        raise FrozenRunHashError("confirmatory run requires freeze_record.json before submission") from exc
    if not verification.ok:
        drifted = ", ".join(verification.drifted_files)
        suffix = f"; drifted_files={drifted}" if drifted else ""
        raise FrozenRunHashError(f"confirmatory run freeze_record.json mismatch: {verification.status}{suffix}")
    frozen_hash = load_frozen_hash(config)
    if frozen_hash is None:
        raise FrozenRunHashError("confirmatory run requires freeze_record.json before submission")
    return frozen_hash


def _make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def _wall_clock_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _SpendTracker:
    def __init__(self, ceiling: float | None) -> None:
        self._ceiling = ceiling
        self._total = 0.0
        self._lock = threading.Lock()

    def add(self, amount: float) -> None:
        with self._lock:
            self._total += amount

    @property
    def total(self) -> float:
        with self._lock:
            return self._total

    def ceiling_exceeded(self) -> bool:
        if self._ceiling is None:
            return False
        with self._lock:
            return self._total >= self._ceiling


def _update_persona_summary(summary: dict[str, int], episode: dict[str, Any]) -> None:
    if episode.get("module") != "A" or episode.get("call_status") != "ok":
        return
    summary["total"] += 1
    rerun_count = int(episode.get("rerun_count", 0) or 0)
    if rerun_count == 1:
        summary["rerun_once"] += 1
    elif rerun_count >= 2:
        summary["double_leak"] += 1


def _log_persona_summary(summary: dict[str, int]) -> None:
    total = summary["total"]
    if total == 0:
        return
    rerun_total = summary["rerun_once"] + summary["double_leak"]
    logger.info(
        "PERSONA RE-RUN SUMMARY: %d re-run once, %d double-leak accepted, %d total Module A episodes. Re-run rate %.1f%%.",
        summary["rerun_once"],
        summary["double_leak"],
        total,
        100.0 * rerun_total / total,
    )


def _chat(client: Any, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
    signature = inspect.signature(client.chat)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_kwargs:
        return client.chat(model, messages, **kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return client.chat(model, messages, **supported)


def _turn(role: str, speaker: str, text: str, *, truncated: bool = False) -> dict[str, Any]:
    turn: dict[str, Any] = {"role": role, "speaker": speaker, "text": text}
    if truncated:
        turn["truncated"] = True
    return turn


def _episode_id(
    scenario_id: str,
    module: str,
    variant_id: str,
    model: str,
    repeat: int,
    *,
    phase: str | None = None,
    human_sample: str | None = None,
) -> str:
    model_slug = model.replace("/", "__").replace(":", "_")
    if phase is not None:
        human = human_sample or _human_sample_for_phase(phase)
        return f"{scenario_id}-{phase}-human_{human}-{module}-{variant_id}-{model_slug}-r{repeat}"
    return f"{scenario_id}-{module}-{variant_id}-{model_slug}-r{repeat}"


def _existing_episode_ids(path: Path, *, retry_missing: bool = False) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if retry_missing and row.get("call_status") == "missing":
                    continue
                ids.add(row["episode_id"])
    return ids


def _missing_episode_ids(path: Path) -> set[str]:
    # The ids already present in the file as recorded MISSING cells. A --retry-missing run
    # replaces exactly these and appends everything else, so the write path can avoid a
    # full-file rewrite for brand-new ids.
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("call_status") == "missing":
                    ids.add(row["episode_id"])
    return ids


def _prompt_dir(config: SliceConfig) -> Path:
    return Path(config.config_root) / "prompts"
