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
This account posts a SMALL SELECTION from a library of tens of thousands of
photos — be picky. The bar is "a stranger scrolling a feed would stop for this,"
NOT "acceptable snapshot." Judge the shot in front of you, not the scene's
potential: a good subject shot poorly is still a weak photo.

Score "quality" 0-10 (subject + framing + light + informational value):
- 9-10 striking: strong close subject, great light or composition — stops the scroll
- 7-8  genuinely good: one clear subject, close enough to see well, deliberately
       framed, decent light; appealing or clearly informative at first glance
- 5-6  mediocre: a real subject but a weak shot — subject too far away or small,
       much dead space, flat light, messy framing, or nothing new to see
- 3-4  weak: mostly empty background (bare sky/road/wall/water/table), subject
       hard to find, low informational value, or a dull view of an ordinary thing
- 0-2  bad: near-empty, blurry, dark, accidental
Typical casual phone snapshots are 4-6. Score honestly — do NOT default to 7.

Automatic caps (apply the LOWEST that fits):
- main subject far away or small in the frame → at most 5
- empty or featureless areas dominate the frame → at most 4
- viewer learns nothing about the place AND the photo isn't beautiful → at most 4
- blurry, badly exposed, or accidental → at most 3
- a bare logo/sign/storefront close-up with no surrounding scene → at most 3
- a selfie where the photographer is the main subject → at most 3

interesting=true ONLY for shots worth a stranger's attention: a clear, close-enough
subject, deliberately framed, AND (visually appealing OR genuinely informative about
how people live, eat, build, or gather here). Food, markets, malls, street life,
scenery, and night views all QUALIFY as subjects — but only WELL-EXECUTED shots of
them pass. When unsure, set interesting=false and score low: a better photo of the
same kind of subject will come along; nothing is lost by skipping this one."""

CAPTION_BLOCK = """── CAPTION (only if interesting=true) ────────────────────────────────────────
Write for an INTERNATIONAL audience — viewers OUTSIDE China who do NOT know
Chinese food, places, or customs. The caption should make them SEE the photo.

- Describe what is actually visible in plain, vivid words: the dish and its main
  components, or the scene and its mood — what the eye sees, not a bare label.
- Assume ZERO knowledge of Chinese terms. If something has no common English
  name (e.g. zongzi, baozi, jianbing, youtiao), DESCRIBE it instead of naming it
  — e.g. not "zongzi" but "sticky rice dumplings wrapped in bamboo leaves". A
  local name may follow in parentheses, AFTER the description.
- One short, warm sentence in the voice of an expat sharing a real insight about China —
  not an ad, brochure, or sweeping generalization. Stay positive and apolitical; don't
  invent prices, numbers, or names. ≤200 chars, no hashtags.
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
  "description": "one factual sentence about what is shown",
  "shows": "one phrase — what an outsider sees or learns about the place here, and why it is worth posting (or 'little of interest' if empty/generic). Decide quality and interesting AFTER this.",
  "quality": <int 0-10 — see QUALITY scoring>,
  "interesting": bool,
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
        "safe": False, "flags": [flag], "quality": 0,
        "interesting": False, "description": "", "caption": "", "reason": "",
    }


def coerce_quality(parsed: dict) -> None:
    """Clamp parsed["quality"] to an int in 0-10, in place. Missing or invalid
    → 0 (fail-closed: an unscored image never clears the posting bar). Shared by
    the single-image and burst judges so both validate identically."""
    try:
        q = int(parsed.get("quality"))
    except (TypeError, ValueError):
        q = 0
    parsed["quality"] = max(0, min(q, 10))


def call_safety_quality(image_path: Path, cfg: Config) -> tuple[dict, float, bool]:
    """Run combined safety + quality check on image_path.

    Returns (result_dict, elapsed_seconds, used_network).
    result_dict always has all 7 keys: safe, flags, quality (int 0-10),
    interesting, description, caption, reason.
    Fails closed: any error returns safe=False, interesting=False, quality=0.
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
    coerce_quality(parsed)
    parsed.setdefault("flags", [])
    parsed.setdefault("description", "")
    parsed.setdefault("reason", "")
    return parsed, elapsed, used_network
