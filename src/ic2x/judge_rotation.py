"""
Visual rotation check using the multi-provider AI client.

Checks whether the prepared JPEG is correctly oriented (right-side up).
Fail-open: any error returns upright=True so the image is never blocked.
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

logger = logging.getLogger("ic2x.judge_rotation")

ROTATION_PROMPT = """Look at this image and determine if it is correctly oriented (right-side up).
Return ONLY valid JSON — no markdown, no explanation:
{"upright": true, "rotate_cw_degrees": 0}

rotate_cw_degrees must be one of: 0, 90, 180, 270
- 0   → already upright, no rotation needed
- 90  → rotate clockwise 90° to make it upright
- 180 → rotate 180° (image is upside down)
- 270 → rotate clockwise 270° to make it upright

JSON only."""

_OK: dict = {"upright": True, "rotate_cw_degrees": 0}


def call_rotation(image_path: Path, cfg: Config) -> tuple[dict, float, bool]:
    """Check if image_path is correctly oriented.

    Returns (result_dict, elapsed_seconds, used_network).
    result_dict always has {"upright": bool, "rotate_cw_degrees": int}.
    Fails open: errors return {"upright": True, "rotate_cw_degrees": 0}.
    """
    model, _ = parse_model_effort(cfg.rotation_model)
    is_ollama = provider_for_model(model) == "ollama"
    max_px = cfg.ollama_image_max_px if is_ollama else cfg.rotation_image_max_px

    parsed, elapsed, ok, used_network = call_vision_judge(
        model_string=cfg.rotation_model,
        ollama_base_url=cfg.ollama_base_url,
        call=JudgeCall(
            image_path=image_path,
            prompt=ROTATION_PROMPT,
            max_px=max_px,
            fail_value=dict(_OK),
            refused_value=dict(_OK),
            label="rotation",
        ),
    )

    if not ok:
        return parsed, elapsed, used_network

    if "upright" not in parsed or "rotate_cw_degrees" not in parsed:
        logger.warning(
            "rotation: unexpected response shape for %s: %s",
            image_path.name, str(parsed)[:200],
        )
        return dict(_OK), elapsed, used_network

    degrees = int(parsed.get("rotate_cw_degrees", 0))
    if degrees not in (0, 90, 180, 270):
        logger.warning(
            "rotation: unexpected degrees=%s for %s — treating as 0",
            degrees, image_path.name,
        )
        degrees = 0
    return {"upright": bool(parsed["upright"]), "rotate_cw_degrees": degrees}, elapsed, used_network
