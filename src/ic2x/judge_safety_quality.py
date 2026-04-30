"""
Safety + quality check in a single AI call.

Combines both checks into one prompt to halve image-upload cost — the image
token is the dominant expense.
Fail-closed: any error returns safe=False, interesting=False.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ic2x.config import Config
from ic2x.utils.ai_client import (
    JudgeCall,
    call_vision_judge,
    parse_model_effort,
    provider_for_model,
)

logger = logging.getLogger("ic2x.judge_safety_quality")

JUDGE_PROMPT = """You are a content reviewer for a personal photo account on X (Twitter).
Analyze this image and return ONLY valid JSON — no markdown, no explanation.

Schema:
{
  "safe": bool,
  "flags": [],
  "interesting": bool,
  "description": "one factual sentence about what is shown",
  "caption": "casual tweet caption, ≤100 chars, no hashtags",
  "reason": "brief explanation of the quality decision"
}

── SAFETY ─────────────────────────────────────────────────────────────────────────────
Flag any that apply (include the string in the "flags" array):
- "nudity_sexual"    — nudity, underwear-only, sexualized content, sexual acts
- "minor_primary"    — person appearing under 18 is the primary subject
- "violence_gore"    — blood, injury, weapons aimed at people, visible death
- "illegal_visible"  — drug use, drug paraphernalia, other clear illegal acts
- "hate_symbols"     — extremist imagery, hate group symbols
- "private_document" — IDs, passports, credit cards, readable license plates,
                       screens showing private messages or personal data
- "privacy_closeup"  — single unrecognised individual, close-up, clearly invasive
- "obscene_other"    — anything grossly offensive not covered above

Set "safe": false if any flag applies. Be strict — when in doubt, flag.
If safe=false: set interesting=false, description="", caption="", reason="unsafe".

── QUALITY (only if safe=true) ───────────────────────────────────────────────────────────────────────────
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


def _fail(flag: str) -> dict:
    return {
        "safe": False, "flags": [flag],
        "interesting": False, "description": "", "caption": "", "reason": "",
    }


def call_safety_quality(image_path: Path, cfg: Config) -> tuple[dict, float, bool]:
    """Run combined safety + quality check on image_path.

    Returns (result_dict, elapsed_seconds, used_network).
    result_dict always has all 6 keys: safe, flags, interesting,
    description, caption, reason.
    Fails closed: any error returns safe=False, interesting=False.
    """
    model, _ = parse_model_effort(cfg.judge_model)
    is_ollama = provider_for_model(model) == "ollama"
    max_px = cfg.ollama_image_max_px if is_ollama else cfg.judge_image_max_px

    parsed, elapsed, ok, used_network = call_vision_judge(
        model_string=cfg.judge_model,
        ollama_base_url=cfg.ollama_base_url,
        call=JudgeCall(
            image_path=image_path,
            prompt=JUDGE_PROMPT,
            max_px=max_px,
            fail_value=_fail("error:no_client"),
            refused_value=_fail("gemini_refused"),
            label="judge",
        ),
    )

    if not ok:
        return parsed, elapsed, used_network

    if "safe" not in parsed or "interesting" not in parsed:
        logger.warning(
            "judge: unexpected response shape for %s: %s",
            image_path.name, str(parsed)[:200],
        )
        return _fail("error:bad_shape"), elapsed, used_network

    caption = parsed.get("caption") or ""
    if len(caption) > 100:
        logger.info(
            "judge: truncated caption from %d to 100 chars for %s",
            len(caption), image_path.name,
        )
    parsed["caption"] = caption[:100]
    parsed.setdefault("flags", [])
    parsed.setdefault("description", "")
    parsed.setdefault("reason", "")
    return parsed, elapsed, used_network
