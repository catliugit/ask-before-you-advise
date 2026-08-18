from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from slice.client import (
    _FALLBACK_PRICE_IN_PER_MTOK,
    _FALLBACK_PRICE_OUT_PER_MTOK,
    ChatOutputError,
    OpenRouterClient,
    _apply_reasoning_policy,
    _payload_reasoning_setting,
)


class StubOpenRouterClient(OpenRouterClient):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(
            cache_dir=tmp_path / "cache",
            cost_log_path=tmp_path / "cost.jsonl",
            api_key="test-key",
        )
        self.posts = 0
        self.payloads = []

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        self.posts += 1
        self.payloads.append(payload)
        return (
            {
                "id": f"raw-{self.posts}",
                "model": "stub/test:served",
                "choices": [{"message": {"content": f"response-{self.posts}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0055},
            },
            f"request-{self.posts}",
        )


def test_cache_scope_splits_identical_messages_and_reuses_same_scope(tmp_path):
    client = StubOpenRouterClient(tmp_path)
    messages = [{"role": "user", "content": "Should I invest this?"}]

    first = client.chat("stub/test", messages, cache_scope="episode-r0", timestamp="t")
    second = client.chat("stub/test", messages, cache_scope="episode-r1", timestamp="t")
    third = client.chat("stub/test", messages, cache_scope="episode-r0", timestamp="t")

    assert first.cached is False
    assert second.cached is False
    assert third.cached is True
    assert third.text == first.text
    assert client.posts == 2
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2


def test_reasoning_policy_changes_openrouter_payload(tmp_path):
    client = StubOpenRouterClient(tmp_path)
    messages = [{"role": "user", "content": "hello"}]

    client.chat("stub/test", messages, reasoning="default", cache_scope="default")
    client.chat("stub/test", messages, reasoning="on", cache_scope="on")
    client.chat("stub/test", messages, reasoning="off", cache_scope="off")
    client.chat("stub/test", messages, reasoning="xhigh", cache_scope="xhigh")

    assert "reasoning" not in client.payloads[0]
    assert client.payloads[1]["reasoning"] == {"enabled": True}
    assert client.payloads[2]["reasoning"] == {"enabled": False}
    assert client.payloads[3]["reasoning"] == {"effort": "xhigh"}
    assert all(payload["usage"] == {"include": True} for payload in client.payloads)


def test_apply_reasoning_policy_supports_effort_levels():
    payload: dict[str, Any] = {}
    _apply_reasoning_policy(payload, "xhigh")
    assert payload["reasoning"] == {"effort": "xhigh"}

    payload = {}
    _apply_reasoning_policy(payload, "on")
    assert payload["reasoning"] == {"enabled": True}

    payload = {}
    _apply_reasoning_policy(payload, "off")
    assert payload["reasoning"] == {"enabled": False}

    payload = {}
    _apply_reasoning_policy(payload, "default")
    assert "reasoning" not in payload


def test_payload_reasoning_setting_round_trips_effort_levels():
    assert _payload_reasoning_setting({"reasoning": {"effort": "xhigh"}}) == "xhigh"
    assert _payload_reasoning_setting({"reasoning": {"effort": "high"}}) == "high"


def test_session_cost_tracks_fresh_api_spend_and_cache_hits_add_zero(tmp_path):
    client = StubOpenRouterClient(tmp_path)
    messages = [{"role": "user", "content": "Should I invest this?"}]

    first = client.chat("stub/test", messages, cache_scope="spend", timestamp="t")
    after_fresh = client.session_cost
    second = client.chat("stub/test", messages, cache_scope="spend", timestamp="t")

    assert first.cached is False
    assert second.cached is True
    assert client.posts == 1
    assert after_fresh == pytest.approx(0.0055)
    assert client.session_cost == pytest.approx(after_fresh)


class SequenceOpenRouterClient(OpenRouterClient):
    def __init__(
        self,
        tmp_path: Path,
        responses: list[Any],
        *,
        bad_output_retries: int = 3,
        retry_max_tokens_cap: int = 16000,
    ) -> None:
        super().__init__(
            cache_dir=tmp_path / "cache",
            cost_log_path=tmp_path / "cost.jsonl",
            api_key="test-key",
            bad_output_retries=bad_output_retries,
            retry_max_tokens_cap=retry_max_tokens_cap,
        )
        self.responses = list(responses)
        self.payloads = []

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        self.payloads.append(payload)
        content = self.responses.pop(0)
        return (
            {
                "id": f"raw-{len(self.payloads)}",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0042},
            },
            f"request-{len(self.payloads)}",
        )


def test_retry_on_empty_content_escalates_then_succeeds(tmp_path):
    client = SequenceOpenRouterClient(tmp_path, [None, "usable"], bad_output_retries=1)

    result = client.chat(
        "stub/test",
        [{"role": "user", "content": "hello"}],
        max_tokens=100,
        reasoning="on",
        role="test_model",
        episode_id="episode-1",
    )

    assert result.text == "usable"
    assert result.retry_count == 1
    assert [payload["max_tokens"] for payload in client.payloads] == [100, 200]
    assert client.payloads[0]["reasoning"] == {"enabled": True}
    assert client.payloads[1]["reasoning"] == {"enabled": True}
    assert all(payload["usage"] == {"include": True} for payload in client.payloads)

    cached = client.chat(
        "stub/test",
        [{"role": "user", "content": "hello"}],
        max_tokens=100,
        reasoning="on",
        role="test_model",
        episode_id="episode-1",
    )

    assert cached.cached is True
    assert cached.retry_count == 1


def test_retry_on_judge_json_parse_failure_escalates_then_succeeds(tmp_path):
    client = SequenceOpenRouterClient(
        tmp_path,
        ["not json", '{"dimensions": []}'],
        bad_output_retries=1,
    )

    result = client.chat(
        "stub/judge",
        [{"role": "user", "content": "judge"}],
        max_tokens=50,
        reasoning="on",
        role="judge_pass1",
        episode_id="episode-json",
        expect_json=True,
        expected_json_keys={"dimensions": list},
    )

    assert result.text == '{"dimensions": []}'
    assert [payload["max_tokens"] for payload in client.payloads] == [50, 100]
    assert all(payload["reasoning"] == {"enabled": True} for payload in client.payloads)
    assert all(payload["usage"] == {"include": True} for payload in client.payloads)


@pytest.mark.parametrize("bad_deferral", [True, 4])
def test_retry_on_strict_deferral_shape_failure_escalates_then_succeeds(tmp_path, bad_deferral):
    client = SequenceOpenRouterClient(
        tmp_path,
        [
            json.dumps({"confidence": 0.9, "safety_flag": "none", "deferral_score": bad_deferral}, separators=(",", ":")),
            '{"confidence":0.9,"safety_flag":"none","deferral_score":3}',
        ],
        bad_output_retries=1,
    )

    result = client.chat(
        "stub/judge",
        [{"role": "user", "content": "judge"}],
        max_tokens=50,
        reasoning="on",
        role="judge_pass2",
        episode_id="episode-deferral",
        expect_json=True,
        expected_json_keys={"confidence": (int, float), "safety_flag": str, "deferral_score": int},
    )

    assert result.text == '{"confidence":0.9,"safety_flag":"none","deferral_score":3}'
    assert result.retry_count == 1
    assert [payload["max_tokens"] for payload in client.payloads] == [50, 100]


def test_session_cost_counts_invalid_output_retry_attempts(tmp_path):
    client = SequenceOpenRouterClient(
        tmp_path,
        ["not json", '{"dimensions": []}'],
        bad_output_retries=1,
    )

    client.chat(
        "stub/judge",
        [{"role": "user", "content": "judge"}],
        role="judge_pass1",
        episode_id="episode-json",
        expect_json=True,
        expected_json_keys={"dimensions": list},
    )

    assert len(client.payloads) == 2
    assert client.session_cost == pytest.approx(0.0084)


def test_retry_exhaustion_raises_clear_chat_output_error(tmp_path):
    client = SequenceOpenRouterClient(tmp_path, ["", ""], bad_output_retries=1)

    with pytest.raises(ChatOutputError) as excinfo:
        client.chat(
            "stub/test",
            [{"role": "user", "content": "hello"}],
            role="test_model",
            episode_id="episode-empty",
        )

    message = str(excinfo.value)
    assert "episode-empty" in message
    assert "test_model" in message
    assert "empty or null content" in message


def test_cost_from_usage_and_model_version_from_response(tmp_path):
    client = StubOpenRouterClient(tmp_path)

    result = client.chat("stub/test", [{"role": "user", "content": "hello"}], cache_scope="cost")

    assert result.cost_estimate == 0.0055
    assert result.model_version == "stub/test:served"
    assert result.usage_include is True
    assert result.sent_model == "stub/test"


def test_usage_include_is_sent_on_every_retry(tmp_path):
    client = SequenceOpenRouterClient(tmp_path, [None, "usable"], bad_output_retries=1)

    client.chat("stub/test", [{"role": "user", "content": "hello"}], cache_scope="retry-usage")

    assert len(client.payloads) == 2
    assert all(payload["usage"] == {"include": True} for payload in client.payloads)


def test_reasoning_off_warning(tmp_path, caplog):
    client = StubOpenRouterClient(tmp_path)

    with caplog.at_level("WARNING", logger="slice.client"):
        client.chat("stub/test", [{"role": "user", "content": "hello"}], reasoning="off", cache_scope="off-warning")

    assert "Sending reasoning=off" in caplog.text


class MissingCostOpenRouterClient(OpenRouterClient):
    """Stub whose responses never carry usage.cost (the audit's silent-under-count case)."""

    def __init__(self, tmp_path: Path, *, usage: dict[str, Any] | None, model_prices: dict[str, tuple[float, float]] | None = None) -> None:
        super().__init__(
            cache_dir=tmp_path / "cache",
            cost_log_path=tmp_path / "cost.jsonl",
            api_key="test-key",
            model_prices=model_prices,
        )
        self.usage = usage

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        raw: dict[str, Any] = {
            "id": "raw-1",
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
        }
        if self.usage is not None:
            raw["usage"] = dict(self.usage)
        return raw, "request-1"


def test_missing_usage_cost_advances_tracked_spend_with_panel_prices(tmp_path, caplog):
    # Binding test for the cost-guard fallback: a response WITHOUT usage.cost must still
    # advance the tracked spend, via token counts x model-panel prices.
    client = MissingCostOpenRouterClient(
        tmp_path,
        usage={"prompt_tokens": 1000, "completion_tokens": 500},
        model_prices={"stub/test": (10.0, 30.0)},
    )

    with caplog.at_level(logging.WARNING, logger="slice.client"):
        result = client.chat("stub/test", [{"role": "user", "content": "hello"}])

    expected = (1000 * 10.0 + 500 * 30.0) / 1_000_000
    assert client.session_cost == pytest.approx(expected)
    assert client.session_cost > 0.0
    assert result.cost_estimate == pytest.approx(expected)
    assert "usage.cost absent" in caplog.text


def test_missing_usage_cost_without_panel_price_uses_conservative_fallback(tmp_path, caplog):
    client = MissingCostOpenRouterClient(
        tmp_path,
        usage={"prompt_tokens": 1000, "completion_tokens": 500},
        model_prices=None,
    )

    with caplog.at_level(logging.WARNING, logger="slice.client"):
        client.chat("stub/test", [{"role": "user", "content": "hello"}])

    expected = (1000 * _FALLBACK_PRICE_IN_PER_MTOK + 500 * _FALLBACK_PRICE_OUT_PER_MTOK) / 1_000_000
    assert client.session_cost == pytest.approx(expected)
    assert client.session_cost > 0.0
    assert "usage.cost absent" in caplog.text
    assert "conservative fallback prices" in caplog.text


def test_missing_usage_block_entirely_still_advances_spend(tmp_path, caplog):
    client = MissingCostOpenRouterClient(tmp_path, usage=None, model_prices={"stub/test": (10.0, 30.0)})

    with caplog.at_level(logging.WARNING, logger="slice.client"):
        client.chat("stub/test", [{"role": "user", "content": "hello"}], max_tokens=100)

    # No token counts at all: ~2 prompt tokens from message length, worst-case 100 completion.
    expected = (2 * 10.0 + 100 * 30.0) / 1_000_000
    assert client.session_cost == pytest.approx(expected)
    assert client.session_cost > 0.0
    assert "token counts absent" in caplog.text


def test_present_usage_cost_still_used_verbatim(tmp_path):
    client = StubOpenRouterClient(tmp_path)

    result = client.chat("stub/test", [{"role": "user", "content": "hello"}], cache_scope="verbatim-cost")

    assert result.cost_estimate == pytest.approx(0.0055)
    assert client.session_cost == pytest.approx(0.0055)


def test_retry_ladder_clamps_max_tokens_at_absolute_cap(tmp_path, caplog):
    # Binding test for the retry-ladder cap: min(base * 2**attempt, cap), logged when it binds.
    client = SequenceOpenRouterClient(
        tmp_path,
        [None, None, None, "usable"],
        bad_output_retries=3,
        retry_max_tokens_cap=6000,
    )

    with caplog.at_level(logging.WARNING, logger="slice.client"):
        result = client.chat(
            "stub/test",
            [{"role": "user", "content": "hello"}],
            max_tokens=4000,
            role="cheap_panel",
            episode_id="episode-cap",
        )

    assert result.text == "usable"
    assert [payload["max_tokens"] for payload in client.payloads] == [4000, 6000, 6000, 6000]
    assert "max_tokens retry ladder clamped" in caplog.text


def test_retry_ladder_default_cap_clamps_worst_rung_only(tmp_path):
    client = SequenceOpenRouterClient(tmp_path, [None, None, None, "usable"], bad_output_retries=3)

    client.chat(
        "stub/test",
        [{"role": "user", "content": "hello"}],
        max_tokens=4000,
        role="cheap_panel",
        episode_id="episode-default-cap",
    )

    assert [payload["max_tokens"] for payload in client.payloads] == [4000, 8000, 16000, 16000]


def test_retry_max_tokens_cap_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        OpenRouterClient(
            cache_dir=tmp_path / "cache",
            cost_log_path=tmp_path / "cost.jsonl",
            api_key="test-key",
            retry_max_tokens_cap=0,
        )
