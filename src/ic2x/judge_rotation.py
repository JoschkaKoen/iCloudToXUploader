"""
Visual rotation check using the multi-provider AI client.

Checks whether the prepared JPEG is correctly oriented (right-side up).
Fail-open: any error returns upright=True so the image is never blocked.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from ic2x.utils.ai_client import (
    make_ai_client,
    build_thinking_kwargs,
    call_ollama_chat,
    collect_streamed_response,
    strip_json_fences,
)
from ic2x.utils.image_utils import encode_image_b64

logger = logging.getLogger("ic2x.judge_rotation")

ROTATION_PROMPT = """Look at this image and determine if it is correctly oriented (right-side up).
Return ONLY valid JSON — no markdown, no explanation:
{"upright": true, "rotate_cw_degrees": 0}

rotate_cw_degrees must be one of: 0, 90, 180, 270
- 0   → already upright, no rotation needed
- 90  → rotate clockwise 90° to make it upright
- 180 → rotate 180° (image is upside down)
- 270 → rotate clockwise 270° to make it upright

JSON only."""


def call_rotation(image_path: Path) -> tuple[dict, float]:
    """Check if image_path is correctly oriented.

    Returns (result_dict, elapsed_seconds).
    result_dict always has {"upright": bool, "rotate_cw_degrees": int}.
    Fails open: errors return {"upright": True, "rotate_cw_degrees": 0}.
    """
    t0 = time.monotonic()
    _ok = {"upright": True, "rotate_cw_degrees": 0}
    try:
        result = make_ai_client(model_env="ROTATION_MODEL")
        if result is None:
            logger.warning("rotation: no AI client available — skipping check")
            return _ok, time.monotonic() - t0

        client, model, provider, effort = result
        use_stream, extra_kwargs = build_thinking_kwargs(provider, effort)

        # Mirror the provider-split pattern used by judge_safety / judge_quality
        if provider == "ollama":
            _raw_px = os.environ.get("OLLAMA_IMAGE_MAX_PX", "").strip()
            _max_px: int | None = int(_raw_px) if _raw_px.isdigit() else None
        else:
            _raw_px = os.environ.get("ROTATION_IMAGE_MAX_PX", "1024").strip()
            _max_px = int(_raw_px) if _raw_px.isdigit() else 1024
        img_b64 = encode_image_b64(image_path, max_px=_max_px)

        if provider == "ollama":
            ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            raw = call_ollama_chat(ollama_base, model, "/no_think\n" + ROTATION_PROMPT, img_b64)
        else:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": ROTATION_PROMPT},
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
                    logger.info("rotation: model refused %s — skipping", image_path.name)
                    return _ok, time.monotonic() - t0
                raw = resp.choices[0].message.content or ""

        parsed = json.loads(strip_json_fences(raw))
        if "upright" not in parsed or "rotate_cw_degrees" not in parsed:
            raise ValueError(f"Unexpected response shape: {raw[:200]}")
        degrees = int(parsed.get("rotate_cw_degrees", 0))
        if degrees not in (0, 90, 180, 270):
            logger.warning("rotation: unexpected degrees=%s for %s — treating as 0",
                           degrees, image_path.name)
            degrees = 0
        return {"upright": bool(parsed["upright"]), "rotate_cw_degrees": degrees}, time.monotonic() - t0

    except Exception as exc:
        logger.warning("rotation: error for %s: %s — skipping", image_path.name, exc)
        return _ok, time.monotonic() - t0
