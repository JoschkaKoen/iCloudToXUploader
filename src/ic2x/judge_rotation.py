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

ROTATION_PROMPT = """Decide the clockwise rotation needed to make this photo upright.

Reason from real-world cues before answering:
- Sky, ceilings, and light sources are usually at the TOP.
- People, buildings, trees, bottles, and signs stand VERTICALLY.
- Faces are right-side up (eyes above mouth).
- Readable text runs horizontally, left-to-right.
- Floors, tables, and water surfaces are horizontal, near the BOTTOM.

Then return ONLY valid JSON — no markdown:
{"upright": <bool>, "rotate_cw_degrees": <0|90|180|270>}

rotate_cw_degrees = the clockwise rotation to APPLY to make it upright:
- 0   → already upright
- 90  → currently the top of the scene points LEFT  → rotate 90° clockwise
- 180 → upside down
- 270 → currently the top of the scene points RIGHT → rotate 270° clockwise (= 90° counter-clockwise)

JSON only."""

_OK: dict = {"upright": True, "rotate_cw_degrees": 0}


def call_rotation(image_path: Path, cfg: Config, model_string: str | None = None
                  ) -> tuple[dict, float, bool]:
    """Check if image_path is correctly oriented.

    Returns (result_dict, elapsed_seconds, used_network).
    result_dict always has {"upright": bool, "rotate_cw_degrees": int}.
    Fails open: errors return {"upright": True, "rotate_cw_degrees": 0}.
    model_string overrides cfg.rotation_model (used by `autorotate` to compare).
    """
    model_str = model_string or cfg.rotation_model
    model, _ = parse_model_effort(model_str)
    is_ollama = provider_for_model(model) == "ollama"
    max_px = cfg.ollama_image_max_px if is_ollama else cfg.rotation_image_max_px

    parsed, elapsed, ok, used_network = call_vision_judge(
        model_string=model_str,
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
