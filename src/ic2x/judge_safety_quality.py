"""
Safety + quality check in a single AI call.

Saves one image upload per processed photo versus calling judge_safety and
judge_quality separately — the image token cost is the dominant expense.
Fail-closed: any error returns safe=False, interesting=False.
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

logger = logging.getLogger("ic2x.judge_safety_quality")

JUDGE_PROMPT = """You are a content reviewer for a personal photo account on X (Twitter).
Analyze this image and return ONLY valid JSON — no markdown, no explanation.

Schema:
{
  "safe": bool,
  "flags": [],
  "interesting": bool,
  "description": "one factual sentence about what is shown",
  "caption": "casual tweet caption, \u2264100 chars, no hashtags",
  "reason": "brief explanation of the quality decision"
}

\u2500\u2500 SAFETY \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
Flag any that apply (include the string in the "flags" array):
- "nudity_sexual"    \u2014 nudity, underwear-only, sexualized content, sexual acts
- "minor_primary"    \u2014 person appearing under 18 is the primary subject
- "violence_gore"    \u2014 blood, injury, weapons aimed at people, visible death
- "illegal_visible"  \u2014 drug use, drug paraphernalia, other clear illegal acts
- "hate_symbols"     \u2014 extremist imagery, hate group symbols
- "private_document" \u2014 IDs, passports, credit cards, readable license plates,
                       screens showing private messages or personal data
- "privacy_closeup"  \u2014 single unrecognised individual, close-up, clearly invasive
- "obscene_other"    \u2014 anything grossly offensive not covered above

Set "safe": false if any flag applies. Be strict \u2014 when in doubt, flag.
If safe=false: set interesting=false, description="", caption="", reason="unsafe".

\u2500\u2500 QUALITY (only if safe=true) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
Would this image be worth posting on a personal photo account on X?

Accept: interesting composition, travel scenes with character, striking light
        or colour, unusual subjects, moments with a visible story, humor,
        nature, architecture.

Reject as NOT interesting:
- selfies or images where the photographer's face/body is the main subject
- blurry, dark, or accidentally triggered shots
- plain sky, plain wall, empty table, unmade bed
- food photos with no distinctive context
- duplicates of extremely common scenes (generic latte art, standard sunset)
- screenshots or screencasts
- nothing visually engaging after 2 seconds of looking

JSON only."""

_ERROR: dict = {
    "safe": False, "flags": [],
    "interesting": False, "description": "", "caption": "", "reason": "",
}


def call_safety_quality(image_path: Path) -> tuple[dict, float]:
    """Run combined safety + quality check on image_path.

    Returns (result_dict, elapsed_seconds).
    result_dict always has all 6 keys: safe, flags, interesting,
    description, caption, reason.
    Fails closed: any error returns safe=False, interesting=False.
    """
    t0 = time.monotonic()
    try:
        result = make_ai_client(model_env="JUDGE_MODEL")
        if result is None:
            logger.warning("judge: no AI client available (check API key / JUDGE_MODEL)")
            return {**_ERROR, "flags": ["error:no_client"]}, time.monotonic() - t0

        client, model, provider, effort = result
        use_stream, extra_kwargs = build_thinking_kwargs(provider, effort)

        if provider == "ollama":
            _raw_px = os.environ.get("OLLAMA_IMAGE_MAX_PX", "").strip()
            _max_px: int | None = int(_raw_px) if _raw_px.isdigit() else None
        else:
            _raw_px = os.environ.get("JUDGE_IMAGE_MAX_PX", "1024").strip()
            _max_px = int(_raw_px) if _raw_px.isdigit() else 1024
        img_b64 = encode_image_b64(image_path, max_px=_max_px)

        if provider == "ollama":
            ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            raw = call_ollama_chat(ollama_base, model, "/no_think\n" + JUDGE_PROMPT, img_b64)
        else:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": JUDGE_PROMPT},
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
                    logger.info("judge: model refused to analyse %s", image_path.name)
                    return {**_ERROR, "flags": ["gemini_refused"]}, time.monotonic() - t0
                raw = resp.choices[0].message.content or ""

        parsed = json.loads(strip_json_fences(raw))
        if "safe" not in parsed or "interesting" not in parsed:
            raise ValueError(f"Unexpected response shape: {raw[:200]}")
        if parsed.get("caption"):
            parsed["caption"] = parsed["caption"][:100]
        parsed.setdefault("flags", [])
        parsed.setdefault("description", "")
        parsed.setdefault("caption", "")
        parsed.setdefault("reason", "")
        return parsed, time.monotonic() - t0

    except Exception as exc:
        logger.warning("judge: error for %s: %s", image_path.name, exc)
        return {**_ERROR, "flags": [f"error:{type(exc).__name__}"]}, time.monotonic() - t0
