# -*- coding: utf-8 -*-
"""Shared LLM client factory.

Provider is inferred automatically from the model name — no separate
AI_PROVIDER setting is needed.

Supported providers
-------------------
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

ollama  (exact model names registered in ``exact_models``)
    base_url : OLLAMA_BASE_URL env var  (default: http://localhost:11434/v1)
               Override with a Tailscale IP to offload inference to another machine.
    api_key  : not required — "ollama" is used automatically as a dummy value
    example  : qwen3-vl:8b
    note     : To add another Ollama model, append its name to ``exact_models``
               in the Ollama ``_ProviderDef`` entry.

Provider routing
----------------
Exact model names (``exact_models``) are checked first across all providers.
Prefix matching (``model_prefixes``) is used as a fallback for cloud providers
whose model families share a common name prefix.

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
OLLAMA_BASE_URL   Optional — Ollama server URL (default: http://localhost:11434/v1)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("ic2x.ai_client")


@dataclass(frozen=True)
class _ProviderDef:
    """Immutable descriptor for a single LLM provider."""
    name: str
    base_url: str
    api_key_env: str
    model_prefixes: tuple[str, ...]   # prefix match against model name (cloud providers)
    exact_models: tuple[str, ...] = ()  # exact model names that route to this provider
    api_key_default: str | None = None  # used when env var is unset (e.g. Ollama dummy key)
    timeout: float | None = None        # per-provider request timeout in seconds


# Registry of known providers.
# Routing: exact_models checked first (across all providers), then model_prefixes.
# To add a new Ollama model, append its name to the Ollama entry's exact_models tuple.
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
    _ProviderDef(
        name="ollama",
        # Default for `make_ai_client` callers that don't pass an explicit
        # ollama_base_url. Production callers should pass cfg.ollama_base_url
        # so the URL is sourced from a single place (config.py).
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        model_prefixes=(),
        exact_models=("qwen3-vl:8b",),
        api_key_default="ollama",  # SDK requires non-empty; Ollama ignores the value
        timeout=120.0,             # cold model load can take 10-30s on first call
    ),
]

# Fallback model when no model env var is set anywhere.
_DEFAULT_MODEL = "gemini-2.5-flash"


def provider_for_model(model: str) -> str:
    """Return the provider name for *model*.

    Exact matches (``exact_models``) are checked first across all providers,
    then prefix matches (``model_prefixes``).  Falls back to ``gemini`` for
    unknown model names.
    """
    m = model.lower()
    # Pass 1: exact match takes absolute priority
    for pdef in _PROVIDER_REGISTRY:
        if m in pdef.exact_models:
            return pdef.name
    # Pass 2: prefix match for cloud provider model families
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
    model_string: str,
    *,
    ollama_base_url: str | None = None,
) -> tuple[Any, str, str, str | None] | None:
    """Return ``(client, model_name, provider, effort)`` or ``None`` if the API key is missing.

    Parameters
    ----------
    model_string:
        Pre-resolved model identifier from Config, optionally with a
        thinking-effort suffix: ``"gemini-2.5-flash, low"``. Callers compose
        this from cfg.judge_model / cfg.rotation_model so this module no
        longer reads model env vars itself.
    ollama_base_url:
        When the resolved provider is ``ollama``, override the registry's
        default base URL with this value. Pass ``cfg.ollama_base_url`` so all
        Ollama traffic goes through one configured endpoint.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None

    model, effort = parse_model_effort(model_string or _DEFAULT_MODEL)
    provider = provider_for_model(model)
    pdef = next((p for p in _PROVIDER_REGISTRY if p.name == provider), _PROVIDER_REGISTRY[0])

    api_key = os.environ.get(pdef.api_key_env, "").strip()
    if not api_key and pdef.name == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key and pdef.api_key_default:
        api_key = pdef.api_key_default
    if not api_key:
        return None

    base_url = ollama_base_url if (pdef.name == "ollama" and ollama_base_url) else pdef.base_url
    try:
        client_kwargs: dict = {"api_key": api_key, "base_url": base_url}
        if pdef.timeout is not None:
            client_kwargs["timeout"] = pdef.timeout
        client = OpenAI(**client_kwargs)
    except Exception:
        return None

    return client, model, provider, effort


def warmup_ollama(base_url: str, model: str, max_wait: float = 90.0) -> None:
    """Block until the Ollama model is loaded into VRAM, or raise on failure.

    Steps:
    1. GET /api/tags — verify the model is pulled (fast, no VRAM needed).
       Raises RuntimeError with an actionable message if missing.
    2. POST /api/generate with an empty prompt — the native Ollama endpoint
       blocks until the model is resident in VRAM, then returns.  This is
       the Ollama-documented way to pre-load a model; it never returns 503.

    Parameters
    ----------
    base_url:
        OpenAI-compatible URL, e.g. ``http://localhost:11434/v1``.
        The native Ollama API root is derived by stripping ``/v1``.
    model:
        Exact Ollama model name, e.g. ``qwen3-vl:8b``.
    max_wait:
        Maximum seconds to wait for the model to load into VRAM.
    """
    import urllib.request
    import json as _json

    api_base = base_url.rstrip("/")
    if api_base.endswith("/v1"):
        api_base = api_base[:-3]

    # Step 1: verify model is pulled
    try:
        with urllib.request.urlopen(f"{api_base}/api/tags", timeout=10) as resp:
            tags = _json.loads(resp.read())
        pulled = [m["name"] for m in tags.get("models", [])]
        model_base = model.split(":")[0]
        if not any(m == model or m.startswith(model_base + ":") for m in pulled):
            raise RuntimeError(
                f"Ollama model '{model}' is not pulled.\n"
                f"Run:  ollama pull {model}"
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {api_base}. Is it running?\n"
            f"Error: {exc}"
        ) from exc

    # Step 2: load model into VRAM via native API — blocks until ready, no 503
    try:
        payload = _json.dumps({"model": model, "prompt": "", "keep_alive": "10m"}).encode()
        req = urllib.request.Request(
            f"{api_base}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=max_wait) as resp:
            resp.read()
    except Exception as exc:
        raise RuntimeError(
            f"Ollama model '{model}' did not load within {max_wait:.0f}s.\n"
            f"Error: {exc}"
        ) from exc


def unload_ollama(base_url: str, model: str) -> None:
    """Immediately unload *model* from Ollama's memory.

    Sends ``keep_alive: 0`` to the native ``/api/generate`` endpoint, which
    is the Ollama-documented way to evict a model from VRAM/RAM.
    Errors are silently ignored — unloading is best-effort.
    """
    import urllib.request
    import json as _json

    api_base = base_url.rstrip("/")
    if api_base.endswith("/v1"):
        api_base = api_base[:-3]

    try:
        payload = _json.dumps({"model": model, "keep_alive": 0}).encode()
        req = urllib.request.Request(
            f"{api_base}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except Exception:
        pass  # best-effort — don't crash the pipeline over a cleanup call


def call_ollama_chat(
    base_url: str,
    model: str,
    prompt: str,
    image_b64: str,
    *,
    timeout: float = 120.0,
) -> str:
    """Call the native Ollama /api/chat endpoint; return raw response text.

    Uses urllib.request only — no httpx, no proxy interference.
    Images go in the ``images`` array (raw base64, no data-URI prefix).
    ``format: "json"`` constrains the model to emit valid JSON.
    """
    import urllib.request
    import json as _json

    api_base = base_url.rstrip("/")
    if api_base.endswith("/v1"):
        api_base = api_base[:-3]

    payload = _json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": False,
        "format": "json",
    }).encode()

    req = urllib.request.Request(
        f"{api_base}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = _json.loads(resp.read())
    return data["message"]["content"]


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


# ─── vision judge dispatch ────────────────────────────────────────────────────


@dataclass
class JudgeCall:
    """Inputs to a single vision-judge call.

    fail_value is returned (with elapsed_s) on any error: missing client,
    network failure, JSON decode failure, schema validation failure.
    refused_value is returned when the model issues a content_filter refusal;
    if None, fail_value is used.
    """
    image_path: Path
    prompt: str
    max_px: int | None
    fail_value: dict
    refused_value: dict | None = None
    label: str = "judge"   # used in log messages, e.g. "judge:" or "rotation:"


def call_vision_judge(
    *,
    model_string: str,
    ollama_base_url: str,
    call: JudgeCall,
) -> tuple[dict, float, bool]:
    """Run a vision JSON judge end-to-end.

    Returns (result_dict, elapsed_seconds, ok). When ok=True the dict is the
    model's parsed JSON — the caller is responsible for schema validation
    and any post-processing. When ok=False the dict is `call.fail_value`
    (any error) or `call.refused_value` (content_filter), already in the
    caller's expected shape; return it directly.
    """
    from ic2x.utils.image_utils import encode_image_b64

    t0 = time.monotonic()
    refused = call.refused_value if call.refused_value is not None else call.fail_value

    try:
        result = make_ai_client(model_string, ollama_base_url=ollama_base_url)
        if result is None:
            logger.warning("%s: no AI client available (check API key / model)", call.label)
            return call.fail_value, time.monotonic() - t0, False

        client, model, provider, effort = result
        use_stream, extra_kwargs = build_thinking_kwargs(provider, effort)
        img_b64 = encode_image_b64(call.image_path, max_px=call.max_px)

        if provider == "ollama":
            raw = call_ollama_chat(
                ollama_base_url, model, "/no_think\n" + call.prompt, img_b64
            )
        else:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": call.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]}]
            if use_stream:
                stream = client.chat.completions.create(
                    model=model, messages=messages, stream=True, **extra_kwargs
                )
                raw = collect_streamed_response(stream)
            else:
                resp = client.chat.completions.create(
                    model=model, messages=messages, stream=False,
                    response_format={"type": "json_object"}, **extra_kwargs
                )
                if resp.choices[0].finish_reason == "content_filter":
                    logger.info("%s: model refused %s", call.label, call.image_path.name)
                    return refused, time.monotonic() - t0, False
                raw = resp.choices[0].message.content or ""

        return json.loads(strip_json_fences(raw)), time.monotonic() - t0, True

    except Exception as exc:
        logger.warning("%s: error for %s: %s", call.label, call.image_path.name, exc)
        return call.fail_value, time.monotonic() - t0, False
