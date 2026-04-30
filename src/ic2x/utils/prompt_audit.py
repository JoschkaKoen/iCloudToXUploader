"""Optional on-disk prompt + response audit (debugging)."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def audit_base_path() -> Path | None:
    raw = os.environ.get("IC2X_AI_PROMPT_LOG_DIR", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def build_audit_prompt_path(
    log_dir: Path,
    image_path: Path,
    label: str,
    model: str,
) -> Path:
    """Unique path under *log_dir* (date shard + safe stem + label + hash)."""
    ident = f"{image_path.resolve()}|{label}|{model}".encode()
    short = hashlib.sha256(ident).hexdigest()[:10]
    safe_stem = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in image_path.stem
    )[:80]
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return log_dir / day / f"{safe_stem}_{label}_{short}.md"


def save_prompt(
    path: Path | None,
    *,
    model: str = "",
    system: str = "",
    messages: list[dict[str, Any]],
) -> None:
    if path is None:
        return
    try:
        sections: list[str] = [f"# Prompt — {model}\n"]
        if system:
            sections.append(f"## system\n\n{system}\n")
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                text_only = "\n".join(texts)
            else:
                text_only = str(content)
            role = msg.get("role", "user")
            sections.append(f"## {role}\n\n{text_only}\n")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(sections), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def save_response(
    prompt_path: Path | None,
    response: str,
    *,
    thinking: str | None = None,
) -> None:
    if prompt_path is None:
        return
    try:
        if thinking:
            body = f"[thinking]\n{thinking}\n[/thinking]\n\n{response}"
        else:
            body = response
        resp_path = prompt_path.with_name(prompt_path.stem + "_response.txt")
        resp_path.parent.mkdir(parents=True, exist_ok=True)
        resp_path.write_text(body, encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
