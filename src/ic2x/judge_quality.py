"""
Quality / interest check using the multi-provider AI client.

Only runs if the image passed the safety check.
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

logger = logging.getLogger("ic2x.judge_quality")

QUALITY_PROMPT = """Would this image be worth posting on a personal photo account on X?
Return ONLY valid JSON — no markdown, no explanation:
{
  "interesting": bool,
  "description": "one factual sentence",
  "caption": "casual, \u2264100 chars, no hashtags",
  "reason": "brief explanation of your decision"
}

Accept: interesting composition, travel scenes with character, striking light or colour,
        unusual subjects, moments with a visible story, humor, nature, architecture.

Reject as NOT interesting:
- selfies or images where the photographer's face/body is the main subject
- blurry, dark, or accidentally triggered shots
- plain sky, plain wall, empty table, unmade bed
- food photos with no distinctive context
- duplicates of extremely common scenes (generic latte art, standard sunset)
- screenshots or screencasts
- nothing visually engaging after 2 seconds of looking

JSON only."""


def call_quality(image_path: Path) -> tuple[dict, float]:
    """Run the quality check on image_path.

    Returns (result_dict, elapsed_seconds).
    result_dict always has {"interesting": bool, "description", "caption", "reason"}.
    """
    t0 = time.monotonic()
    _error_result = {"interesting": False, "description": "", "caption": "", "reason": ""}
    try:
        result = make_ai_client(model_env="QUALITY_MODEL")
        if result is None:
            logger.warning("quality: no AI client available (check GEMINI_API_KEY / QUALITY_MODEL)")
            _error_result["reason"] = "error:no_client"
            return _error_result, time.monotonic() - t0

        client, model, provider, effort = result
        use_stream, extra_kwargs = build_thinking_kwargs(provider, effort)

        prompt = QUALITY_PROMPT
        if provider == "ollama":
            _raw_px = os.environ.get("OLLAMA_IMAGE_MAX_PX", "").strip()
            _max_px: int | None = int(_raw_px) if _raw_px.isdigit() else None
        else:
            _raw_px = os.environ.get("QUALITY_IMAGE_MAX_PX", "1024").strip()
            _max_px = int(_raw_px) if _raw_px.isdigit() else 1024
        img_b64 = encode_image_b64(image_path, max_px=_max_px)

        if provider == "ollama":
            ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            raw = call_ollama_chat(ollama_base, model, "/no_think\n" + prompt, img_b64)
        else:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
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
                    logger.info("quality: model refused to analyse %s", image_path.name)
                    _error_result["reason"] = "gemini_refused"
                    return _error_result, time.monotonic() - t0
                raw = resp.choices[0].message.content or ""

        parsed = json.loads(strip_json_fences(raw))
        if "interesting" not in parsed:
            raise ValueError(f"Unexpected response shape: {raw[:200]}")
        if parsed.get("caption"):
            parsed["caption"] = parsed["caption"][:100]
        return parsed, time.monotonic() - t0

    except Exception as exc:
        logger.warning("quality: error for %s: %s", image_path.name, exc)
        _error_result["reason"] = f"error:{type(exc).__name__}"
        return _error_result, time.monotonic() - t0
