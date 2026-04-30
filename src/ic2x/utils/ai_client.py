# -*- coding: utf-8 -*-
"""Shared LLM client factory.

Provider is inferred automatically from the model name — no separate
AI_PROVIDER setting is needed.

Supported providers
-------------------
gemini  (model names starting with ``gemini``)
    base_url : https://generativelanguage.googleapis.com/v1beta/openai/
    api_key  : GEMINI_API_KEY  (GOOGLE_API_KEY accepted as fallback)

xai  (model names starting with ``grok``)
    base_url : https://api.x.ai/v1
    api_key  : XAI_API_KEY

qwen  (model names starting with ``qwen``)
    base_url : https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key  : DASHSCOPE_API_KEY

kimi / moonshot  (prefix ``kimi`` or ``moonshot``)
    base_url : https://api.moonshot.cn/v1  (override with KIMI_BASE_URL for .ai)
    api_key  : KIMI_API_KEY

ollama  (exact model names registered in ``exact_models``)
    base_url : OLLAMA_BASE_URL / cfg.ollama_base_url

Per-call-type model overrides use optional effort: off | low | high after a comma.

Determinism (optional): ALL_AI_TEMPERATURE, ALL_AI_SEED — injected on chat.completions
when not set per-call (skipped for kimi-k2* temperature — fixed server-side).

Environment variables (API keys)
---------------------------------
GEMINI_API_KEY, GOOGLE_API_KEY, XAI_API_KEY, DASHSCOPE_API_KEY, KIMI_API_KEY
KIMI_BASE_URL  Optional Moonshot API root
IC2X_AI_RESPONSE_CACHE, IC2X_AI_RESPONSE_CACHE_DIR, IC2X_AI_RESPONSE_CACHE_OLLAMA
IC2X_AI_PROMPT_LOG_DIR, IC2X_AI_RETRY_MAX_ATTEMPTS, IC2X_AI_RETRY_BASE_SLEEP
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ic2x.utils.api_retry import ContentFilterRefusal, retry_api_call
from ic2x.utils.prompt_audit import (
    audit_base_path,
    build_audit_prompt_path,
    save_prompt,
    save_response,
)
from ic2x.utils.response_cache import (
    cache_get,
    cache_put,
    ollama_cache_override_enabled,
    response_cache_enabled,
    vision_cache_key,
)
from ic2x.utils.response_parsing import parse_json_safe, strip_json_fences

logger = logging.getLogger("ic2x.ai_client")

# ---------------------------------------------------------------------------
# Thread-safe token usage
# ---------------------------------------------------------------------------

_usage_lock = threading.Lock()
_run_usage: dict[str, dict[str, int]] = {}


def record_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
) -> None:
    with _usage_lock:
        e = _run_usage.setdefault(model, {"input": 0, "output": 0, "thinking": 0})
        e["input"] += input_tokens
        e["output"] += output_tokens
        e["thinking"] += thinking_tokens


def _extract_reasoning_tokens(u: Any) -> int:
    details = getattr(u, "completion_tokens_details", None)
    nested = (getattr(details, "reasoning_tokens", 0) if details else 0) or 0
    if nested:
        return nested
    return getattr(u, "reasoning_tokens", 0) or 0


def get_run_usage() -> dict[str, dict[str, int]]:
    with _usage_lock:
        return {m: dict(v) for m, v in _run_usage.items()}


def reset_run_usage() -> None:
    with _usage_lock:
        _run_usage.clear()


_run_call_stats: dict[str, dict[str, float]] = {}


def record_call(model: str, duration_s: float) -> None:
    """Optional latency metric bucket (not used in cost report)."""
    with _usage_lock:
        e = _run_call_stats.setdefault(model, {"calls": 0.0, "total_duration_s": 0.0})
        e["calls"] += 1
        e["total_duration_s"] += duration_s


def get_run_call_stats() -> dict[str, dict[str, float]]:
    with _usage_lock:
        return {m: dict(v) for m, v in _run_call_stats.items()}


def reset_run_call_stats() -> None:
    with _usage_lock:
        _run_call_stats.clear()


def _read_default_temperature() -> float | None:
    raw = os.environ.get("ALL_AI_TEMPERATURE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _read_default_seed() -> int | None:
    raw = os.environ.get("ALL_AI_SEED", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class _ProviderDef:
    """Immutable descriptor for a single LLM provider."""
    name: str
    base_url: str
    api_key_env: str
    model_prefixes: tuple[str, ...]
    exact_models: tuple[str, ...] = ()
    api_key_default: str | None = None
    timeout: float | None = None


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
        name="kimi",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="KIMI_API_KEY",
        model_prefixes=("kimi", "moonshot"),
    ),
    _ProviderDef(
        name="ollama",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        model_prefixes=(),
        exact_models=("qwen3-vl:8b",),
        api_key_default="ollama",
        timeout=120.0,
    ),
]

_DEFAULT_MODEL = "gemini-2.5-flash"


def provider_for_model(model: str) -> str:
    m = model.lower()
    for pdef in _PROVIDER_REGISTRY:
        if m in pdef.exact_models:
            return pdef.name
    for pdef in _PROVIDER_REGISTRY:
        if any(m.startswith(pfx) for pfx in pdef.model_prefixes):
            return pdef.name
    return "gemini"


def parse_model_effort(value: str) -> tuple[str, str | None]:
    if "," in value:
        model_part, effort_part = value.split(",", 1)
        effort = effort_part.strip().lower() or None
        if effort not in ("off", "low", "high"):
            effort = None
        return model_part.strip(), effort
    return value.strip(), None


def build_thinking_kwargs(provider: str, effort: str | None) -> tuple[bool, dict]:
    if provider == "gemini":
        if effort == "off":
            return False, {"reasoning_effort": "none"}
        if effort in ("low", "high"):
            return True, {"reasoning_effort": effort}
        return True, {}

    if provider == "qwen":
        if effort == "off":
            return False, {"extra_body": {"enable_thinking": False}}
        return True, {"extra_body": {"enable_thinking": True}}

    if provider == "kimi":
        if effort in ("low", "high"):
            return True, {"extra_body": {"thinking": {"type": "enabled"}}}
        return False, {"extra_body": {"thinking": {"type": "disabled"}}}

    return False, {}


class _UsageTrackingStream:
    def __init__(self, stream: Any, model: str, t0: float | None = None) -> None:
        self._stream = stream
        self._model = model
        self._t0 = t0
        self._recorded = False

    def __iter__(self):
        last_usage = None
        try:
            for chunk in self._stream:
                u = getattr(chunk, "usage", None)
                if u is not None:
                    last_usage = u
                yield chunk
        finally:
            if last_usage is not None and self._model and not self._recorded:
                record_usage(
                    self._model,
                    getattr(last_usage, "prompt_tokens", 0) or 0,
                    getattr(last_usage, "completion_tokens", 0) or 0,
                    _extract_reasoning_tokens(last_usage),
                )
                if self._t0 is not None:
                    record_call(self._model, time.perf_counter() - self._t0)
                self._recorded = True

    def __enter__(self) -> "_UsageTrackingStream":
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*args)


class _TrackedCompletions:
    def __init__(self, completions: Any, model: str, deterministic: bool = True) -> None:
        self._c = completions
        self._model = model
        self._deterministic = deterministic

    def create(self, *args: Any, **kwargs: Any) -> Any:
        inject = self._deterministic and not self._model.startswith("kimi-k2")
        if inject:
            if "temperature" not in kwargs:
                t = _read_default_temperature()
                if t is not None:
                    kwargs["temperature"] = t
            if "seed" not in kwargs:
                s = _read_default_seed()
                if s is not None and not self._model.startswith("kimi-k2"):
                    kwargs["seed"] = s
        is_stream = kwargs.get("stream", False)
        if is_stream:
            opts = dict(kwargs.get("stream_options") or {})
            opts.setdefault("include_usage", True)
            kwargs["stream_options"] = opts
        t0 = time.perf_counter()
        resp = self._c.create(*args, **kwargs)
        if is_stream:
            return _UsageTrackingStream(resp, self._model, t0)
        u = getattr(resp, "usage", None)
        if u:
            record_usage(
                self._model,
                getattr(u, "prompt_tokens", 0) or 0,
                getattr(u, "completion_tokens", 0) or 0,
                _extract_reasoning_tokens(u),
            )
            record_call(self._model, time.perf_counter() - t0)
        return resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._c, name)


class _TrackedChat:
    def __init__(self, chat: Any, model: str, deterministic: bool = True) -> None:
        self._chat = chat
        self.completions = _TrackedCompletions(chat.completions, model, deterministic=deterministic)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _TrackedOpenAIClient:
    def __init__(self, client: Any, model: str, deterministic: bool = True) -> None:
        self._client = client
        self.chat = _TrackedChat(client.chat, model, deterministic=deterministic)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def make_ai_client(
    model_string: str,
    *,
    ollama_base_url: str | None = None,
    deterministic: bool = True,
) -> tuple[Any, str, str, str | None] | None:
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

    base_url = pdef.base_url
    if pdef.name == "ollama" and ollama_base_url:
        base_url = ollama_base_url
    elif pdef.name == "kimi":
        base_url = os.environ.get("KIMI_BASE_URL", "").strip() or base_url

    try:
        client_kwargs: dict = {"api_key": api_key, "base_url": base_url}
        if pdef.timeout is not None:
            client_kwargs["timeout"] = pdef.timeout
        raw_client = OpenAI(**client_kwargs)
    except Exception:
        return None

    client: Any = _TrackedOpenAIClient(raw_client, model, deterministic=deterministic)
    return client, model, provider, effort


def warmup_ollama(base_url: str, model: str, max_wait: float = 90.0) -> None:
    import urllib.request
    import json as _json

    api_base = base_url.rstrip("/")
    if api_base.endswith("/v1"):
        api_base = api_base[:-3]

    try:
        with urllib.request.urlopen(f"{api_base}/api/tags", timeout=10) as resp:
            tags = _json.loads(resp.read())
        pulled = [m["name"] for m in tags.get("models", [])]
        model_base = model.split(":")[0]
        if not any(m == model or m.startswith(model_base + ":") for m in pulled):
            raise RuntimeError(
                f"Ollama model '{model}' is not pulled.\nRun:  ollama pull {model}"
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {api_base}. Is it running?\nError: {exc}"
        ) from exc

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
            f"Ollama model '{model}' did not load within {max_wait:.0f}s.\nError: {exc}"
        ) from exc


def unload_ollama(base_url: str, model: str) -> None:
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
        pass


def call_ollama_chat(
    base_url: str,
    model: str,
    prompt: str,
    image_b64: str,
    *,
    timeout: float = 120.0,
) -> str:
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


def collect_streamed_response(stream: Any) -> str:
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
        print()
    return "".join(content_parts).strip()


@dataclass
class JudgeCall:
    image_path: Path
    prompt: str
    max_px: int | None
    fail_value: dict
    refused_value: dict | None = None
    label: str = "judge"


def call_vision_judge(
    *,
    model_string: str,
    ollama_base_url: str,
    call: JudgeCall,
) -> tuple[dict, float, bool, bool]:
    """Returns (result_dict, elapsed_seconds, ok, used_network)."""
    from ic2x.utils.image_utils import encode_image_b64

    t0 = time.monotonic()
    refused = call.refused_value if call.refused_value is not None else call.fail_value
    used_network = False

    def _elapsed() -> float:
        return time.monotonic() - t0

    try:
        result = make_ai_client(model_string, ollama_base_url=ollama_base_url)
        if result is None:
            logger.warning("%s: no AI client available (check API key / model)", call.label)
            return call.fail_value, _elapsed(), False, False

        client, model, provider, effort = result
        use_stream, extra_kwargs = build_thinking_kwargs(provider, effort)
        img_b64 = encode_image_b64(call.image_path, max_px=call.max_px)

        cache_on = response_cache_enabled()
        if provider == "ollama" and not ollama_cache_override_enabled():
            cache_on = False

        ckey: str | None = None
        if cache_on:
            ckey = vision_cache_key(model=model, user_prompt=call.prompt, image_b64=img_b64)
            hit = cache_get(ckey)
            if hit and isinstance(hit.get("response"), str):
                raw = hit["response"]
                used_network = False
                audit_path: Path | None = None
                log_base = audit_base_path()
                if log_base is not None:
                    audit_path = build_audit_prompt_path(
                        log_base, call.image_path, call.label, model
                    )
                    messages = [{"role": "user", "content": [
                        {"type": "text", "text": call.prompt},
                        {"type": "text", "text": "[image omitted — from cache]"},
                    ]}]
                    save_prompt(audit_path, model=model, messages=messages)
                    save_response(audit_path, raw + "\n\n[ic2x: served from response cache]")
                parsed = parse_json_safe(raw)
                if parsed is None:
                    excerpt = raw[:240].replace("\n", " ") + ("…" if len(raw) > 240 else "")
                    logger.warning(
                        "%s: JSON parse failed (cached) for %s — %s",
                        call.label, call.image_path.name, excerpt,
                    )
                    return call.fail_value, _elapsed(), False, False
                return parsed, _elapsed(), True, used_network

        messages = [{"role": "user", "content": [
            {"type": "text", "text": call.prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]}]

        audit_path = None
        log_base = audit_base_path()
        if log_base is not None:
            audit_path = build_audit_prompt_path(log_base, call.image_path, call.label, model)
            save_prompt(audit_path, model=model, messages=messages)

        raw: str

        if provider == "ollama":

            used_network = True

            def _ollama_once() -> str:
                return call_ollama_chat(
                    ollama_base_url, model, "/no_think\n" + call.prompt, img_b64
                )

            raw = retry_api_call(_ollama_once, label=f"{call.label}:ollama")
        else:
            used_network = True

            def _cloud_once() -> str:
                if use_stream:
                    stream = client.chat.completions.create(
                        model=model, messages=messages, stream=True, **extra_kwargs
                    )
                    return collect_streamed_response(stream)
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    response_format={"type": "json_object"},
                    **extra_kwargs,
                )
                if resp.choices[0].finish_reason == "content_filter":
                    raise ContentFilterRefusal("content_filter")
                return resp.choices[0].message.content or ""

            raw = retry_api_call(_cloud_once, label=f"{call.label}:chat")

        save_response(audit_path, raw)

        parsed = parse_json_safe(raw)
        if parsed is None:
            excerpt = raw[:240].replace("\n", " ") + ("…" if len(raw) > 240 else "")
            logger.warning(
                "%s: JSON parse failed for %s — %s",
                call.label, call.image_path.name, excerpt,
            )
            return call.fail_value, _elapsed(), False, used_network

        if cache_on and ckey is not None and used_network:
            cache_put(ckey, model=model, response=raw)

        return parsed, _elapsed(), True, used_network

    except ContentFilterRefusal:
        logger.info("%s: model refused %s", call.label, call.image_path.name)
        return refused, _elapsed(), False, used_network
    except Exception as exc:
        logger.warning("%s: error for %s: %s", call.label, call.image_path.name, exc)
        return call.fail_value, _elapsed(), False, used_network


__all__ = [
    "JudgeCall",
    "build_thinking_kwargs",
    "call_ollama_chat",
    "call_vision_judge",
    "collect_streamed_response",
    "get_run_usage",
    "make_ai_client",
    "parse_model_effort",
    "print_streamed_response",
    "provider_for_model",
    "record_usage",
    "reset_run_call_stats",
    "reset_run_usage",
    "strip_json_fences",
    "unload_ollama",
    "warmup_ollama",
]
