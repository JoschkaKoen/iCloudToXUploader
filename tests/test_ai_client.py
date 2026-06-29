"""
Offline tests for ai_client's pure model-routing logic — provider detection from a
model name, the "model[, thinking][, max_tokens]" spec parser, and the per-provider
thinking/streaming kwargs. No network, no API keys, no client construction.

Run: .venv/bin/python tests/test_ai_client.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x.utils.ai_client import (  # noqa: E402
    build_thinking_kwargs,
    parse_model_effort,
    parse_model_spec,
    provider_for_model,
)


# ── provider_for_model ───────────────────────────────────────────────────────

def test_provider_detection_by_prefix():
    cases = {
        "gemini-2.5-flash-lite": "gemini",
        "gemini-3.5-flash": "gemini",
        "grok-4": "xai",
        "qwen3.7-plus": "qwen",
        "qwen3-vl-flash": "qwen",
        "kimi-k2.6": "kimi",
        "moonshot-v1-128k": "kimi",
    }
    for model, provider in cases.items():
        assert provider_for_model(model) == provider, model


def test_provider_exact_ollama_model():
    assert provider_for_model("qwen3-vl:8b") == "ollama"   # exact_models wins over qwen prefix


def test_provider_unknown_defaults_to_gemini():
    assert provider_for_model("some-new-model") == "gemini"


def test_provider_is_case_insensitive():
    assert provider_for_model("QWEN3.7-Plus") == "qwen"


# ── parse_model_spec / parse_model_effort ────────────────────────────────────

def test_parse_full_three_position_spec():
    assert parse_model_spec("qwen3.7-plus, 1000, 2000") == ("qwen3.7-plus", "1000", 2000)


def test_parse_effort_word_only():
    assert parse_model_spec("gemini-2.5-flash, off") == ("gemini-2.5-flash", "off", None)
    assert parse_model_spec("grok-4, high") == ("grok-4", "high", None)


def test_parse_bare_model():
    assert parse_model_spec("qwen3.5-flash") == ("qwen3.5-flash", None, None)


def test_parse_ignores_invalid_effort_token():
    # a non-{off,low,high,digit} second field is not an effort
    assert parse_model_spec("model, banana") == ("model", None, None)


def test_parse_model_effort_backcompat():
    assert parse_model_effort("qwen3.7-plus, 1000, 2000") == ("qwen3.7-plus", "1000")


# ── build_thinking_kwargs ────────────────────────────────────────────────────

def test_gemini_thinking_kwargs():
    assert build_thinking_kwargs("gemini", "off") == (False, {"reasoning_effort": "none"})
    assert build_thinking_kwargs("gemini", "low") == (True, {"reasoning_effort": "low"})
    assert build_thinking_kwargs("gemini", None) == (True, {})


def test_qwen_thinking_kwargs_and_budget():
    assert build_thinking_kwargs("qwen", "off") == (False, {"extra_body": {"enable_thinking": False}})
    stream, kw = build_thinking_kwargs("qwen", "1500")
    assert stream is True and kw["extra_body"] == {"enable_thinking": True, "thinking_budget": 1500}


def test_kimi_thinking_kwargs():
    assert build_thinking_kwargs("kimi", None) == (
        False, {"extra_body": {"thinking": {"type": "disabled"}}})
    assert build_thinking_kwargs("kimi", "high") == (
        True, {"extra_body": {"thinking": {"type": "enabled"}}})


def test_max_tokens_is_injected():
    _stream, kw = build_thinking_kwargs("qwen", "off", 2000)
    assert kw["max_tokens"] == 2000


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback; print(f"FAIL {t.__name__}: {exc}"); traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
