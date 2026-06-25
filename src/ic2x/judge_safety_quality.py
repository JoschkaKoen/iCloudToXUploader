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

# Shared criteria — imported by judge_burst.py so single- and multi-image
# judging apply identical safety/quality rules. Keep these the single source.
SAFETY_BLOCK = """── SAFETY ─────────────────────────────────────────────────────────────────────────────
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

Set "safe": false if any flag applies. Be strict — when in doubt, flag."""

QUALITY_BLOCK = """── QUALITY (only if safe=true) ───────────────────────────────────────────────
Would this be a reasonable post on a CASUAL personal photo feed (travel, food,
daily life)? The bar is "nice enough to share," NOT "portfolio-worthy." An
ordinary photo with a clear subject or a pleasant scene IS interesting — it does
not need to be striking or unusual.

POST (interesting=true) — everyday is fine:
- food and meals: dishes, restaurants, street food, drinks (ordinary meals count)
- scenery: rivers, lakes, water reflections, sunsets, golden hour, sky, nature, parks
- places & street scenes with something to look at: shops, night/city views,
  architecture, a sign that is part of a scene, a moment with character or humor

SKIP (interesting=false) — only when GENUINELY weak:
- no clear subject: empty, cluttered, or "vibes" frames with nothing to look at
- a bare logo, plain signboard, or storefront with no surrounding scene
- generic indoor shots with nothing of interest
- blurry, dark, or accidental shots; plain wall / sky / empty table; unmade bed
- a selfie where the photographer is the main subject
- low-information images a stranger would instantly scroll past

When unsure about an ordinary-but-pleasant photo that has a clear subject, POST it.
Reserve interesting=false for photos that are clearly weak or have nothing to show."""

CAPTION_BLOCK = """── CAPTION (only if interesting=true) ────────────────────────────────────────
Write for an INTERNATIONAL audience — viewers OUTSIDE China who do NOT know
Chinese food, places, or customs. The caption should make them SEE the photo.

- Describe what is actually visible in plain, vivid words: the dish and its main
  components, or the scene and its mood — what the eye sees, not a bare label.
- Assume ZERO knowledge of Chinese terms. If something has no common English
  name (e.g. zongzi, baozi, jianbing, youtiao), DESCRIBE it instead of naming it
  — e.g. not "zongzi" but "sticky rice dumplings wrapped in bamboo leaves". A
  local name may follow in parentheses, AFTER the description.
- One casual, warm sentence, like a person sharing their day. ≤200 chars, no hashtags.
- Include at least ONE emoji that MATCHES the real subject (🍜 noodles, 🥢 meal,
  🍲 stew, 🫓 flatbread, 🍳 eggs, 🌅 sunset, 🌊 river, 🏞 landscape, 🏮 lanterns,
  🌃 night city, 🛕 temple, 🍵 tea). Pick it from the photo and VARY it shot to
  shot. NEVER default to the 🇨🇳 flag — use a flag only if a flag is the subject."""

JUDGE_PROMPT = f"""You are a content reviewer for a personal photo account on X (Twitter).
Analyze this image and return ONLY valid JSON — no markdown, no explanation.

Schema:
{{
  "safe": bool,
  "flags": [],
  "interesting": bool,
  "description": "one factual sentence about what is shown",
  "caption": "descriptive caption for an international audience — see CAPTION rules; ≤200 chars, no hashtags, ≥1 topic-matching emoji",
  "reason": "brief explanation of the quality decision"
}}

{SAFETY_BLOCK}
If safe=false: set interesting=false, description="", caption="", reason="unsafe".

{QUALITY_BLOCK}

{CAPTION_BLOCK}

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
    if len(caption) > 200:
        logger.info(
            "judge: truncated caption from %d to 200 chars for %s",
            len(caption), image_path.name,
        )
    parsed["caption"] = caption[:200]
    parsed.setdefault("flags", [])
    parsed.setdefault("description", "")
    parsed.setdefault("reason", "")
    return parsed, elapsed, used_network
