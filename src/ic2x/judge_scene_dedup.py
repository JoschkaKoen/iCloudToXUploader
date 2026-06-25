"""
Cheap "same-scene" dedup: one low-res multi-image VLM call decides whether the
winner about to be posted is the SAME physical scene/object as any recent post —
robust to orientation (portrait vs landscape), crop, or a tiny angle difference,
which pixel pHash cannot catch. Owner rule: two DIFFERENT things of the same kind
(two noodle bowls, two sunsets) are NOT a match.

Fails OPEN: any error returns "not a duplicate" so a post is never blocked.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ic2x.config import Config
from ic2x.utils.ai_client import (
    MultiJudgeCall,
    call_vision_judge_multi,
    parse_model_effort,
    provider_for_model,
)

logger = logging.getLogger("ic2x.scene_dedup")

# {n} markers are string-replaced (not .format) so the JSON braces stay literal.
_PROMPT_TEMPLATE = """You deduplicate a personal photo feed. Image index 0 is a NEW candidate
about to be posted. Images index 1..{n} are photos ALREADY posted recently.

Decide: is image 0 the SAME specific scene or object as any of images 1..{n} —
the same physical place, dish, building, view, or thing — even if it is rotated,
cropped, zoomed, shot portrait vs landscape, or from a slightly different angle
or moment?

Two DIFFERENT things of the same KIND are NOT a match: two different bowls of
noodles, two different sunsets, two different streets are all fine to post.
Only flag a match when it is plausibly the SAME individual subject re-posted.

Return ONLY valid JSON, no markdown:
{"duplicate_of": <the index 1..{n} of the matching posted image, or 0 if none>, "reason": "one short phrase"}"""

_OK = {"duplicate_of": 0, "reason": ""}   # fail-OPEN sentinel: not a duplicate


def call_scene_dedup(candidate: Path, recent_thumbs: list[Path], cfg: Config,
                     model_string: str | None = None) -> tuple[int | None, bool]:
    """Compare `candidate` against `recent_thumbs` (newest-first). Returns
    (dup_index_0based | None, used_network); dup_index indexes INTO recent_thumbs.
    Fails OPEN: empty list / any error / refusal / cloud-less model → (None, used)."""
    if not recent_thumbs:
        return None, False
    model_str = model_string or cfg.scene_dedup_model
    model, _ = parse_model_effort(model_str)
    if provider_for_model(model) == "ollama":
        logger.warning("scene_dedup: cloud model required, got %s — allowing post", model)
        return None, False

    n = len(recent_thumbs)
    parsed, _el, ok, used = call_vision_judge_multi(
        model_string=model_str,
        ollama_base_url=cfg.ollama_base_url,
        call=MultiJudgeCall(
            image_paths=[candidate, *recent_thumbs],
            prompt=_PROMPT_TEMPLATE.replace("{n}", str(n)),
            max_px=cfg.scene_dedup_image_max_px,
            fail_value=dict(_OK),
            refused_value=dict(_OK),
            label="scene_dedup",
        ),
    )
    if not ok:
        return None, used
    try:
        d = int(parsed.get("duplicate_of", 0))
    except (TypeError, ValueError):
        return None, used
    if 1 <= d <= n:
        return d - 1, used      # candidate is image 0; recent posts are 1..n
    return None, used
