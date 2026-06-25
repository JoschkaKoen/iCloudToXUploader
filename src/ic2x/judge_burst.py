"""
Burst judge: one VLM call over N near-identical thumbnails that picks the single
best shot AND vets it (safety + quality + caption). Reuses the exact safety and
quality criteria from judge_safety_quality so single- and multi-image decisions
stay consistent. Fail-closed: any error returns safe=False, interesting=False.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ic2x.config import Config
from ic2x.judge_safety_quality import CAPTION_BLOCK, QUALITY_BLOCK, SAFETY_BLOCK
from ic2x.utils.ai_client import (
    MultiJudgeCall,
    call_vision_judge_multi,
    parse_model_effort,
    provider_for_model,
)

logger = logging.getLogger("ic2x.judge_burst")

_BURST_PROMPT_HEAD = f"""You are choosing ONE photo to post to a personal photo account on X (Twitter),
from a set of near-identical consecutive shots, and vetting it.
You are given N images, each preceded by a line "Image index i:" (i from 0).
Return ONLY valid JSON — no markdown, no explanation.

Schema:
{{
  "best_index": <int 0..N-1 — the single best shot: sharpest, best framed and exposed>,
  "safe": bool,
  "flags": [],
  "interesting": bool,
  "caption": "descriptive caption for the chosen shot for an international audience — see CAPTION rules; ≤200 chars, no hashtags, ≥1 topic-matching emoji",
  "reason": "one short sentence: why this index, and the quality/safety call"
}}

Apply the rules below to the image you choose as best_index.

{SAFETY_BLOCK}
If safe=false: set interesting=false, caption="", reason="unsafe".

{QUALITY_BLOCK}

{CAPTION_BLOCK}"""


def burst_prompt(cfg: Config) -> str:
    """The burst judge prompt, with the owner's do-not-post rules appended."""
    parts = [_BURST_PROMPT_HEAD]
    extra = (cfg.judge_extra_rules or "").strip()
    if extra:
        parts.append(
            "── OWNER'S DO-NOT-POST RULES (these OVERRIDE the above; obey them) ──\n"
            + extra
        )
    parts.append("JSON only.")
    return "\n\n".join(parts)


def _fail(flag: str) -> dict:
    return {
        "best_index": 0, "safe": False, "flags": [flag],
        "interesting": False, "caption": "", "reason": "",
    }


def judge_burst(
    thumb_paths: list[Path], cfg: Config, model_string: str | None = None,
    usage_out: dict | None = None,
) -> tuple[dict, float, bool]:
    """Run the burst judge over thumb_paths (index order == list order).

    Returns (result_dict, elapsed_seconds, used_network). result_dict always has
    best_index (validated into range), safe, flags, interesting, caption (≤100),
    reason. Fails closed. If usage_out is given, the call's {input, output} token
    counts are written into it (per-call cost attribution for parallel runs).
    """
    if not thumb_paths:
        return _fail("error:empty"), 0.0, False

    model_str = model_string or cfg.judge_model
    model, _ = parse_model_effort(model_str)
    if provider_for_model(model) == "ollama":
        logger.warning("judge_burst: cloud model required, got %s", model)
        return _fail("error:ollama_unsupported"), 0.0, False

    parsed, elapsed, ok, used_network = call_vision_judge_multi(
        model_string=model_str,
        ollama_base_url=cfg.ollama_base_url,
        call=MultiJudgeCall(
            image_paths=thumb_paths,
            prompt=burst_prompt(cfg),
            max_px=cfg.judge_image_max_px,
            fail_value=_fail("error:no_client"),
            refused_value=_fail("model_refused"),
            label="burst",
        ),
        usage_out=usage_out,
    )
    if not ok:
        return parsed, elapsed, used_network

    if "safe" not in parsed or "interesting" not in parsed or "best_index" not in parsed:
        logger.warning("judge_burst: unexpected response shape: %s", str(parsed)[:200])
        return _fail("error:bad_shape"), elapsed, used_network

    # Clamp best_index into range — never let a bad index pick a non-existent image.
    try:
        idx = int(parsed["best_index"])
    except (TypeError, ValueError):
        idx = 0
    parsed["best_index"] = max(0, min(idx, len(thumb_paths) - 1))

    caption = parsed.get("caption") or ""
    parsed["caption"] = caption[:200]
    parsed.setdefault("flags", [])
    parsed.setdefault("reason", "")
    return parsed, elapsed, used_network
