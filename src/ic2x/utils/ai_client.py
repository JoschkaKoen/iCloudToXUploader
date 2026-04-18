# -*- coding: utf-8 -*-
"""Shared LLM client factory.

Provider is inferred automatically from the model name — no separate
AI_PROVIDER setting is needed.

Supported providers (auto-detected by model name prefix)
---------------------------------------------------------
gemini  (model names starting with ``gemini``)
    base_url : https://generativelanguage.googleapis.com/v1beta/openai/
    api_key  : GEMINI_API_KEY  (GOOGLE_API_KEY accepted as fallback)
    example  : gemini-2.5-flash, gemini-2.0-flash

xai  (model names starting with ``grok``)
    base_url : https://api.x.ai/v1
    api_key  : XAI_API_KEY
    example  : grok-4-1-fast-non-reasoning, grok-3

    qwen  (model names starting with ``qwen``)
    base_url : https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key  : DASHSCOPE_API_KEY
    example  : qwen3.6-plus, qwen3-32b
    note     : Thinking on → streaming required; thinking off → non-streaming.

Per-call-type model overrides
------------------------------
Each env var accepts an optional thinking-effort suffix after a comma:

    AI_DEFAULT_MODEL=gemini-2.5-flash          # model only (provider default thinking)
    NL_MODEL=gemini-2.5-flash, low             # model + effort
    AI_PRECHECK_MODEL=gemini-2.5-flash-lite, off

Accepted effort values:  off | low | high  (omit = provider default)

    AI_DEFAULT_MODEL   fallback model (and effort) for all calls
    NL_MODEL           prompt interpretation  (overrides AI_DEFAULT_MODEL)
    MCQ_MODEL          AI explanation generation (overrides AI_DEFAULT_MODEL)

Environment variables (API keys)
---------------------------------
GEMINI_API_KEY    Required for gemini models  (GOOGLE_API_KEY accepted as fallback)
XAI_API_KEY       Required for grok models
DASHSCOPE_API_KEY Required for qwen models
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _ProviderDef:
    """Immutable descriptor for a single LLM provider."""
    name: str
    base_url: str
    api_key_env: str
    model_prefixes: tuple[str, ...]  # first match against model name prefix wins


# Registry of known providers. To add a new provider, append one entry here.
_PROVIDER_REGISTRY: list[_ProviderDef] = [
    _ProviderDef(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        model_prefixes=("gemini",),
    ),
    _ProviderDef(
        name="xai",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        model_prefixes=("grok",),
    ),
    _ProviderDef(
        name="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        model_prefixes=("qwen",),
    ),
]

# Fallback model when no model env var is set anywhere.
_DEFAULT_MODEL = "gemini-2.5-flash"


def provider_for_model(model: str) -> str:
    """Return the provider name for *model* based on its name prefix.

    Falls back to ``gemini`` for unknown model names.
    """
    m = model.lower()
    for pdef in _PROVIDER_REGISTRY:
        if any(m.startswith(pfx) for pfx in pdef.model_prefixes):
            return pdef.name
    return "gemini"


def parse_model_effort(value: str) -> tuple[str, str | None]:
    """Split ``"model-name, effort"`` into ``(model, effort)``.

    If no comma is present, effort is ``None`` (provider default).
    Accepted effort values: ``"off"``, ``"low"``, ``"high"``.
    """
    if "," in value:
        model_part, effort_part = value.split(",", 1)
        effort = effort_part.strip().lower() or None
        if effort not in ("off", "low", "high"):
            effort = None
        return model_part.strip(), effort
    return value.strip(), None


def build_thinking_kwargs(provider: str, effort: str | None) -> tuple[bool, dict]:
    """Return ``(use_stream, extra_kwargs)`` for ``client.chat.completions.create()``.

    The caller should pass ``**extra_kwargs`` to ``create()`` and, when
    ``use_stream`` is True, consume the response with
    ``collect_streamed_response()`` instead of reading ``message.content``.

    Effort mapping
    --------------
    Gemini  — ``reasoning_effort="none/low/high"`` top-level param.
              ``off`` maps to ``"none"``.  ``None`` = provider default (no param).
              Streams when thinking is active so output is visible live.
    Qwen    — ``extra_body={"enable_thinking": True/False}`` + streaming when on.
              ``off`` disables thinking and switches to non-streaming mode.
    Grok    — effort is silently ignored; always non-streaming.
    """
    if provider == "gemini":
        if effort == "off":
            return False, {"reasoning_effort": "none"}
        if effort in ("low", "high"):
            # Stream so thinking + content are visible live in the terminal
            return True, {"reasoning_effort": effort}
        # effort is None (provider default) — stream to show live output
        return True, {}

    if provider == "qwen":
        if effort == "off":
            return False, {"extra_body": {"enable_thinking": False}}
        return True, {"extra_body": {"enable_thinking": True}}

    # grok or unknown — no thinking params
    return False, {}


def make_ai_client(
    *,
    model_env: str = "AI_DEFAULT_MODEL",
    legacy_model_env: str = "XAI_MODEL",
    default_model: str | None = None,
) -> tuple[Any, str, str, str | None] | None:
    """Return ``(client, model_name, provider, effort)`` or ``None`` if the API key is missing.

    Parameters
    ----------
    model_env:
        Primary env var for the model (e.g. ``"NL_MODEL"``).  May contain a
        thinking-effort suffix: ``"gemini-2.5-flash, low"``.
    legacy_model_env:
        Fallback env var if *model_env* is unset (e.g. ``"AI_DEFAULT_MODEL"``).
    default_model:
        Model string (optionally with effort suffix) when neither env var is set.
        Defaults to ``AI_DEFAULT_MODEL`` → ``_DEFAULT_MODEL``.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None

    raw = (
        os.environ.get(model_env, "").strip()
        or os.environ.get(legacy_model_env, "").strip()
        or default_model
        or os.environ.get("AI_DEFAULT_MODEL", "").strip()
        or _DEFAULT_MODEL
    )

    model, effort = parse_model_effort(raw)
    provider = provider_for_model(model)
    pdef = next((p for p in _PROVIDER_REGISTRY if p.name == provider), _PROVIDER_REGISTRY[0])

    api_key = os.environ.get(pdef.api_key_env, "").strip()
    if not api_key and pdef.name == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key, base_url=pdef.base_url)
    except Exception:
        return None

    return client, model, provider, effort


def strip_json_fences(raw: str) -> str:
    """Remove markdown code fences that some models add despite being told not to.

    Handles ```json ... ```, ``` ... ```, and leading/trailing whitespace.
    Falls back to extracting the first balanced { ... } block when prose surrounds the JSON.
    """
    import re
    s = raw.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", s)
    if fence:
        return fence.group(1).strip()
    # Stack-walk to find the first balanced { … } so we don't greedily span
    # across multiple top-level JSON objects in the same response.
    start = s.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(s[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return s


def get_api_key_env_name(provider: str | None = None) -> str:
    """Return the env var name for the given provider's API key.

    If *provider* is None, returns the key env for the default model's provider.
    """
    p = provider if provider else provider_for_model(
        os.environ.get("AI_DEFAULT_MODEL", "").strip() or _DEFAULT_MODEL
    )
    pdef = next((pd for pd in _PROVIDER_REGISTRY if pd.name == p), _PROVIDER_REGISTRY[0])
    return pdef.api_key_env


def collect_streamed_response(stream: Any) -> str:
    """Consume a streaming chat completion and return the answer text.

    Skips ``delta.reasoning_content`` (the thinking/scratchpad) and
    accumulates only ``delta.content`` (the final answer).  Works for any
    provider that returns a streaming completion, but is specifically
    designed for Qwen's thinking-mode responses.
    """
    parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            parts.append(delta.content)
    return "".join(parts).strip()


def print_streamed_response(
    stream: Any,
    *,
    print_thinking: bool = True,
    stream_thinking: bool = True,
    print_content: bool = True,
    indent: str = "  ",
    thinking_out: list | None = None,
) -> str:
    """Consume a streaming chat completion, print thinking + content live, return content.

    Thinking (``delta.reasoning_content``) is wrapped in ``[thinking]`` /
    ``[/thinking]`` blocks.  Content (``delta.content``) is printed as-is.
    Only ``delta.content`` is accumulated and returned.

    *print_thinking* controls whether the ``[thinking]`` markers are shown.
    *stream_thinking* controls whether the actual thinking token text is streamed;
    when False the markers still appear but the content is silent.
    If *thinking_out* is a list, thinking text is appended to it regardless.
    """
    content_parts: list[str] = []
    in_thinking = False
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        thinking_text = getattr(delta, "reasoning_content", None) or ""
        content_text = delta.content or ""

        if thinking_text:
            if thinking_out is not None:
                thinking_out.append(thinking_text)
            if print_thinking:
                if not in_thinking:
                    print(f"\n{indent}[thinking]", flush=True)
                    in_thinking = True
                if stream_thinking:
                    print(thinking_text, end="", flush=True)

        if content_text:
            if in_thinking:
                print(f"\n{indent}[/thinking]", flush=True)
                in_thinking = False
            if print_content:
                print(content_text, end="", flush=True)
            content_parts.append(content_text)

    if in_thinking:
        print(f"\n{indent}[/thinking]", flush=True)
    if print_content and content_parts:
        print()  # trailing newline after content
    return "".join(content_parts).strip()
