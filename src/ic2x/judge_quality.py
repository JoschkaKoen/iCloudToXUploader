"""
Gemini quality / interest check.

Only runs if the image passed the safety check.
The prompt is intentionally tunable — iterate after the first week of data.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from PIL import Image

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


def call_quality(image_path: Path, client, model_name: str) -> tuple[dict, float]:
    """
    Run the Gemini quality check on image_path.
    Returns (result_dict, elapsed_seconds).
    result_dict always has {"interesting": bool, "description", "caption", "reason"}.
    """
    from google.genai import types

    t0 = time.monotonic()
    try:
        with Image.open(image_path) as img:
            img_copy = img.copy()

        resp = client.models.generate_content(
            model=model_name,
            contents=[QUALITY_PROMPT, img_copy],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        candidate = resp.candidates[0] if resp.candidates else None
        if candidate and candidate.finish_reason.name == "SAFETY":
            logger.info("quality: Gemini refused to analyse %s", image_path.name)
            return {
                "interesting": False,
                "description": "",
                "caption": "",
                "reason": "gemini_refused",
            }, time.monotonic() - t0

        result = json.loads(resp.text)
        if "interesting" not in result:
            raise ValueError(f"Unexpected response shape: {resp.text[:200]}")
        # Truncate caption to 100 chars just in case
        if result.get("caption"):
            result["caption"] = result["caption"][:100]
        return result, time.monotonic() - t0

    except Exception as exc:
        logger.warning("quality: error for %s: %s", image_path.name, exc)
        return {
            "interesting": False,
            "description": "",
            "caption": "",
            "reason": f"error:{type(exc).__name__}",
        }, time.monotonic() - t0
