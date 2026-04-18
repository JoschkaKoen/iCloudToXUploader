"""
Gemini safety check.

Fail closed: any error (network, quota, parse failure, Gemini refusal)
rejects the image. Never fail open.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from PIL import Image

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


def call_safety(image_path: Path, client, model_name: str) -> tuple[dict, float]:
    """
    Run the Gemini safety check on image_path.
    Returns (result_dict, elapsed_seconds).
    result_dict always has {"safe": bool, "flags": list}.
    """
    from google.genai import types

    t0 = time.monotonic()
    try:
        with Image.open(image_path) as img:
            img_copy = img.copy()

        resp = client.models.generate_content(
            model=model_name,
            contents=[SAFETY_PROMPT, img_copy],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        candidate = resp.candidates[0] if resp.candidates else None
        if candidate and candidate.finish_reason.name == "SAFETY":
            logger.info("safety: Gemini refused to analyse %s", image_path.name)
            return {"safe": False, "flags": ["gemini_refused"]}, time.monotonic() - t0

        result = json.loads(resp.text)
        if "safe" not in result:
            raise ValueError(f"Unexpected response shape: {resp.text[:200]}")
        return result, time.monotonic() - t0

    except Exception as exc:
        logger.warning("safety: error for %s: %s", image_path.name, exc)
        return {"safe": False, "flags": [f"error:{type(exc).__name__}"]}, time.monotonic() - t0
