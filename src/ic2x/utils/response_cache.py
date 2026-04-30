"""Optional disk cache for vision judge responses."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

_DEFAULT_ROOT = Path.home() / ".cache" / "ic2x" / "responses"
_write_lock = threading.Lock()


def response_cache_enabled() -> bool:
    return os.environ.get("IC2X_AI_RESPONSE_CACHE", "").strip().lower() in (
        "1", "true", "yes",
    )


def ollama_cache_override_enabled() -> bool:
    return os.environ.get("IC2X_AI_RESPONSE_CACHE_OLLAMA", "").strip().lower() in (
        "1", "true", "yes",
    )


def cache_root() -> Path:
    override = os.environ.get("IC2X_AI_RESPONSE_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return _DEFAULT_ROOT


def vision_cache_key(*, model: str, user_prompt: str, image_b64: str) -> str:
    h = hashlib.sha256()
    h.update(b"model=")
    h.update(model.encode("utf-8"))
    h.update(b"\0user=")
    h.update(user_prompt.encode("utf-8"))
    h.update(b"\0img_sha256=")
    h.update(hashlib.sha256(image_b64.encode("ascii")).hexdigest().encode("ascii"))
    h.update(b"\0")
    return h.hexdigest()


def _path_for(key: str) -> Path:
    return cache_root() / key[:2] / f"{key}.json"


def cache_get(key: str) -> dict[str, Any] | None:
    path = _path_for(key)
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def cache_put(
    key: str,
    *,
    model: str,
    response: str,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
) -> None:
    path = _path_for(key)
    payload: dict[str, Any] = {
        "key": key,
        "model": model,
        "ts_written": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "response": response,
    }
    if tokens_in is not None:
        payload["tokens_in"] = tokens_in
    if tokens_out is not None:
        payload["tokens_out"] = tokens_out
    try:
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
    except OSError:
        pass
