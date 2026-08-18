from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from tenacity import retry, retry_if_exception, stop_after_delay, wait_exponential

from ._jsonl import append_jsonl


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ReasoningPolicy = Literal["default", "on", "off", "minimal", "low", "medium", "high", "xhigh"]
_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}
# Spend-guard fallback prices, deliberately above every model_panel.yaml entry so a missing
# price can only over-count against the cost ceilings, never under-count.
_FALLBACK_PRICE_IN_PER_MTOK = 10.0
_FALLBACK_PRICE_OUT_PER_MTOK = 50.0
_APPROX_CHARS_PER_TOKEN = 4

logger = logging.getLogger(__name__)

_CACHE_LOCKS_LOCK = threading.Lock()
_CACHE_LOCKS: dict[Path, threading.Lock] = {}


@dataclass(frozen=True)
class ChatResult:
    text: str
    raw: dict[str, Any]
    usage: dict[str, int]
    cost_estimate: float
    request_id: str | None
    model: str
    latency_ms: float
    cached: bool = False
    finish_reason: str | None = None
    model_version: str | None = None
    sent_reasoning: ReasoningPolicy = "default"
    sent_temperature: float | None = None
    usage_include: bool = True
    sent_model: str | None = None
    retry_count: int = 0


class ChatOutputError(RuntimeError):
    def __init__(self, *, episode_id: str | None, role: str | None, reason: str) -> None:
        self.episode_id = episode_id
        self.role = role
        self.reason = reason
        context = []
        if episode_id:
            context.append(f"episode_id={episode_id}")
        if role:
            context.append(f"role={role}")
        context_text = " ".join(context) if context else "unknown call"
        super().__init__(f"OpenRouter chat output invalid after retries ({context_text}): {reason}")


def _is_retryable(exc: BaseException) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    return exc.response.status_code == 429 or exc.response.status_code >= 500


class OpenRouterClient:
    def __init__(
        self,
        cache_dir: str | Path,
        cost_log_path: str | Path,
        *,
        api_key: str | None = None,
        base_url: str = OPENROUTER_URL,
        timeout: float = 60.0,
        bad_output_retries: int = 3,
        model_prices: dict[str, tuple[float, float]] | None = None,
        retry_max_tokens_cap: int = 16000,
    ) -> None:
        if retry_max_tokens_cap < 1:
            raise ValueError("retry_max_tokens_cap must be >= 1")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cost_log_path = Path(cost_log_path)
        self.cost_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self.bad_output_retries = bad_output_retries
        self.model_prices = dict(model_prices) if model_prices else {}
        self.retry_max_tokens_cap = retry_max_tokens_cap
        self._session_cost = 0.0
        self._session_cost_lock = threading.Lock()

    @property
    def session_cost(self) -> float:
        """Cumulative new API spend for this client; cache hits add $0 and score uses a fresh client."""

        with self._session_cost_lock:
            return self._session_cost

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int = 2048,
        cache_scope: str | None = None,
        timestamp: str | None = None,
        reasoning: ReasoningPolicy | None = None,
        role: str | None = None,
        episode_id: str | None = None,
        expect_json: bool = False,
        expected_json_keys: dict[str, type | tuple[type, ...]] | None = None,
        bad_output_retries: int | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if temperature is not None:
            payload["temperature"] = temperature
        _apply_reasoning_policy(payload, reasoning)

        cache_key = self._cache_key(payload, cache_scope=cache_scope)
        cache_path = self.cache_dir / f"{cache_key}.json"
        with _cache_lock_for(cache_path):
            if cache_path.exists():
                cached = json.loads(cache_path.read_text())
                cached["cached"] = True
                cached.setdefault("model_version", cached.get("raw", {}).get("model"))
                cached.setdefault("sent_reasoning", _payload_reasoning_setting(payload))
                cached.setdefault("sent_temperature", payload.get("temperature"))
                cached.setdefault("usage_include", payload.get("usage", {}).get("include") is True)
                cached.setdefault("sent_model", payload.get("model"))
                cached.setdefault("retry_count", 0)
                result = ChatResult(**cached)
                invalid_reason = _invalid_content_reason(
                    result.text,
                    expect_json=expect_json,
                    expected_json_keys=expected_json_keys,
                )
                if invalid_reason is None:
                    self._append_cost_log(result, timestamp=timestamp)
                    return result
                logger.warning(
                    "Ignoring invalid cached chat output for episode_id=%s role=%s: %s",
                    episode_id,
                    role,
                    invalid_reason,
                )
            else:
                result = None

            if not self.api_key:
                raise RuntimeError("OPENROUTER_API_KEY is required for non-cached model calls")

            result = self._post_with_output_retries(
                payload,
                role=role,
                episode_id=episode_id,
                expect_json=expect_json,
                expected_json_keys=expected_json_keys,
                bad_output_retries=bad_output_retries,
            )
            self._write_cache(cache_path, result)
        self._append_cost_log(result, timestamp=timestamp)
        return result

    def _post_with_output_retries(
        self,
        payload: dict[str, Any],
        *,
        role: str | None,
        episode_id: str | None,
        expect_json: bool,
        expected_json_keys: dict[str, type | tuple[type, ...]] | None,
        bad_output_retries: int | None,
    ) -> ChatResult:
        retries = self.bad_output_retries if bad_output_retries is None else bad_output_retries
        if retries < 0:
            raise ValueError("bad_output_retries must be >= 0")

        last_reason = "unknown validation failure"
        for attempt in range(retries + 1):
            attempt_payload = _retry_payload(payload, attempt, max_tokens_cap=self.retry_max_tokens_cap)
            if attempt > 0:
                logger.warning(
                    "Retry attempt %d/%d for episode_id=%s role=%s: doubling max_tokens to %d, reasoning unchanged",
                    attempt,
                    retries,
                    episode_id,
                    role,
                    attempt_payload["max_tokens"],
                )
            if attempt_payload.get("reasoning", {}).get("enabled") is False:
                logger.warning(
                    "Sending reasoning=off for model=%s episode_id=%s role=%s; verify this is intentional",
                    attempt_payload["model"],
                    episode_id,
                    role,
                )
            start = time.monotonic()
            raw, request_id = self._post(attempt_payload)
            accounted_cost = self._account_post_spend(raw, attempt_payload)
            latency_ms = (time.monotonic() - start) * 1000.0
            text = _extract_content(raw)
            invalid_reason = _invalid_content_reason(
                text,
                expect_json=expect_json,
                expected_json_keys=expected_json_keys,
            )
            if invalid_reason is None:
                usage = {
                    "prompt_tokens": int(raw.get("usage", {}).get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(raw.get("usage", {}).get("completion_tokens", 0) or 0),
                }
                if raw.get("usage", {}).get("reasoning_tokens") is not None:
                    usage["reasoning_tokens"] = int(raw.get("usage", {}).get("reasoning_tokens", 0) or 0)
                # accounted_cost is usage.cost when present, else the logged token-price
                # estimate, so the runner's episode-summed ceiling sees the same spend the
                # session guard does.
                cost_estimate = accounted_cost
                return ChatResult(
                    text=str(text).strip(),
                    raw=raw,
                    usage=usage,
                    cost_estimate=cost_estimate,
                    request_id=request_id or raw.get("id"),
                    model=attempt_payload["model"],
                    latency_ms=latency_ms,
                    cached=False,
                    finish_reason=_extract_finish_reason(raw),
                    model_version=raw.get("model") or None,
                    sent_reasoning=_payload_reasoning_setting(attempt_payload),
                    sent_temperature=attempt_payload.get("temperature"),
                    usage_include=attempt_payload.get("usage", {}).get("include") is True,
                    sent_model=attempt_payload["model"],
                    retry_count=attempt,
                )

            last_reason = invalid_reason
            if attempt < retries:
                logger.warning(
                    "Retrying invalid chat output for episode_id=%s role=%s attempt=%s/%s: %s",
                    episode_id,
                    role,
                    attempt + 1,
                    retries,
                    invalid_reason,
                )

        error = ChatOutputError(episode_id=episode_id, role=role, reason=last_reason)
        logger.error(str(error))
        raise error

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_delay(30),
        reraise=True,
    )
    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json(), response.headers.get("x-request-id")

    def _cache_key(self, payload: dict[str, Any], *, cache_scope: str | None = None) -> str:
        cache_material = dict(payload)
        if cache_scope is not None:
            cache_material["_cache_scope"] = cache_scope
        canonical = json.dumps(cache_material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _write_cache(self, cache_path: Path, result: ChatResult) -> None:
        tmp_path: Path | None = None
        fd, tmp_name = tempfile.mkstemp(
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            text=True,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _append_cost_log(self, result: ChatResult, *, timestamp: str | None) -> None:
        record = {
            "timestamp": timestamp,
            "model": result.model,
            "prompt_tokens": result.usage.get("prompt_tokens", 0),
            "completion_tokens": result.usage.get("completion_tokens", 0),
            "cost_estimate": result.cost_estimate,
            "cached": result.cached,
            "request_id": result.request_id,
        }
        append_jsonl(self.cost_log_path, record)

    def _account_post_spend(self, raw: dict[str, Any], payload: dict[str, Any] | None = None) -> float:
        cost = raw.get("usage", {}).get("cost")
        if cost is None:
            cost = self._estimate_spend(raw, payload or {})
            logger.warning(
                "OpenRouter usage.cost absent during spend accounting; counting token-price estimate $%.6f for model=%s",
                cost,
                (payload or {}).get("model"),
            )
        amount = float(cost or 0.0)
        with self._session_cost_lock:
            self._session_cost += amount
        return amount

    def _estimate_spend(self, raw: dict[str, Any], payload: dict[str, Any]) -> float:
        usage = raw.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if prompt_tokens <= 0 and completion_tokens <= 0:
            # No token counts either: assume the worst the request allowed, so the
            # ceilings can only over-count, never silently under-count.
            messages = payload.get("messages") or []
            prompt_tokens = 1 + sum(len(str(message.get("content") or "")) for message in messages) // _APPROX_CHARS_PER_TOKEN
            completion_tokens = int(payload.get("max_tokens") or 0)
            logger.warning(
                "OpenRouter usage token counts absent; conservatively assuming ~%d prompt and %d completion tokens",
                prompt_tokens,
                completion_tokens,
            )
        prices = self.model_prices.get(str(payload.get("model")))
        if prices is None:
            price_in, price_out = _FALLBACK_PRICE_IN_PER_MTOK, _FALLBACK_PRICE_OUT_PER_MTOK
            logger.warning(
                "No model-panel price for model=%s; estimating with conservative fallback prices $%s in / $%s out per MTok",
                payload.get("model"),
                price_in,
                price_out,
            )
        else:
            price_in, price_out = float(prices[0]), float(prices[1])
        return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000.0


def _cache_lock_for(path: Path) -> threading.Lock:
    key = path.resolve()
    with _CACHE_LOCKS_LOCK:
        lock = _CACHE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CACHE_LOCKS[key] = lock
        return lock


def _apply_reasoning_policy(payload: dict[str, Any], policy: ReasoningPolicy | None) -> None:
    if policy in (None, "default"):
        return
    if policy == "on":
        payload["reasoning"] = {"enabled": True}
        return
    if policy == "off":
        payload["reasoning"] = {"enabled": False}
        return
    if policy in _REASONING_EFFORTS:
        payload["reasoning"] = {"effort": policy}
        return
    raise ValueError(f"Unknown reasoning policy: {policy}")


def _retry_payload(payload: dict[str, Any], attempt: int, *, max_tokens_cap: int | None = None) -> dict[str, Any]:
    retry_payload = dict(payload)
    base = int(payload["max_tokens"])
    requested = base * (2**attempt)
    if max_tokens_cap is not None and requested > max_tokens_cap:
        logger.warning(
            "max_tokens retry ladder clamped: attempt %d requested %d, absolute per-call cap is %d",
            attempt,
            requested,
            max_tokens_cap,
        )
        requested = max_tokens_cap
    if attempt > 0 or requested != base:
        retry_payload["max_tokens"] = requested
    return retry_payload


def _payload_reasoning_setting(payload: dict[str, Any]) -> ReasoningPolicy:
    reasoning = payload.get("reasoning", {})
    enabled = reasoning.get("enabled")
    if enabled is True:
        return "on"
    if enabled is False:
        return "off"
    effort = reasoning.get("effort")
    if effort is not None:
        return str(effort)  # type: ignore[return-value]
    return "default"


def _extract_content(raw: dict[str, Any]) -> str | None:
    try:
        content = raw["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    if content is None:
        return None
    return str(content)


def _extract_finish_reason(raw: dict[str, Any]) -> str | None:
    try:
        finish_reason = raw["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return str(finish_reason) if finish_reason is not None else None


def _invalid_content_reason(
    text: str | None,
    *,
    expect_json: bool,
    expected_json_keys: dict[str, type | tuple[type, ...]] | None,
) -> str | None:
    if text is None or not str(text).strip():
        return "empty or null content"
    if not expect_json:
        return None
    try:
        parsed = _loads_json_object(str(text))
    except ValueError as exc:
        return f"JSON parse failure: {exc}"
    if expected_json_keys:
        for key, expected_type in expected_json_keys.items():
            if key not in parsed:
                return f"JSON shape failure: missing key '{key}'"
            if key == "deferral_score":
                value = parsed[key]
                if isinstance(value, bool) or not isinstance(value, int):
                    return "JSON shape failure: key 'deferral_score' has wrong type"
                if not 0 <= value <= 3:
                    return "JSON shape failure: key 'deferral_score' out of range"
                continue
            if not isinstance(parsed[key], expected_type):
                return f"JSON shape failure: key '{key}' has wrong type"
    return None


def _loads_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError("no JSON object found")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise ValueError("JSON root is not an object")
    return parsed
