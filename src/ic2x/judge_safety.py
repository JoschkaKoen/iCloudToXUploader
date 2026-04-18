"""
Safety check using the multi-provider AI client.

Fail closed: any error (network, quota, parse failure, refusal) rejects the image.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ic2x.utils.ai_client import (
    make_ai_client,
    build_thinking_kwargs,
    collect_streamed_response,
    strip_json_fences,
)
from ic2x.utils.image_utils import encode_image_b64

logger = logging.getLogger("ic2x.judge_safety")

SAFETY_PROMPT = """You are a strict content safety reviewer. Analyze this image and return ONLY valid JSON — no markdown, no explanation.

Schema: {"safe": bool, "flags": []}

Flag any that apply (include the string in the array):
- "nudity_sexual"     — nudity, underwear-only, sexualized content, sexual acts
- "minor_primary"     — person appearing under 18 is the primary subject of the image
- "violence_gore"     — blood, injury, weapons aimed at people, visible death
- "illegal_visible"   — drug use, drug paraphernalia, other clear illegal acts
- "hate_symbols"      — extremist imagery, hate group symbols
- "private_document"  — IDs, passports, credit cards, readable license plates, screens showing private messages or personal data
- "privacy_closeup"   — single unrecognised individual, close-up, clearly invasive
- "obscene_other"     — anything grossly offensive not covered above

Be strict. When in doubt, flag. Return {"safe": false, "flags": ["flag_name"]} to reject.
JSON only."""


def call_safety(image_path: Path) -> tuple[dict, float]:
    """Run the safety check on image_path.

    Returns (result_dict, elapsed_seconds).
    result_dict always has {"safe": bool, "flags": list}.
    """
    t0 = time.monotonic()
    try:
        result = make_ai_client(model_env="SAFETY_MODEL")
        if result is None:
            logger.warning("safety: no AI client available (check GEMINI_API_KEY / SAFETY_MODEL)")
            return {"safe": False, "flags": ["error:no_client"]}, time.monotonic() - t0

        client, model, provider, effort = result
        use_stream, extra_kwargs = build_thinking_kwargs(provider, effort)

        img_b64 = encode_image_b64(image_path)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": SAFETY_PROMPT},
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
                logger.info("safety: model refused to analyse %s", image_path.name)
                return {"safe": False, "flags": ["gemini_refused"]}, time.monotonic() - t0
            raw = resp.choices[0].message.content or ""

        parsed = json.loads(strip_json_fences(raw))
        if "safe" not in parsed:
            raise ValueError(f"Unexpected response shape: {raw[:200]}")
        return parsed, time.monotonic() - t0

    except Exception as exc:
        logger.warning("safety: error for %s: %s", image_path.name, exc)
        return {"safe": False, "flags": [f"error:{type(exc).__name__}"]}, time.monotonic() - t0
