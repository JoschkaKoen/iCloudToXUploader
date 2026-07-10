"""
Visual rotation check using the multi-provider AI client.

Two methods:
- call_rotation_single: one image, "estimate the rotation" (legacy — VLMs are weak
  at mental rotation, especially 180°).
- call_rotation: pick-4 — all 4 rotations of the photo in ONE multi-image call,
  shuffled, and the model picks which one is upright. Comparison beats estimation;
  a confident=false escape hatch keeps top-down/ambiguous shots untouched.

Fail-open: any error returns upright=True so the image is never blocked.
"""

from __future__ import annotations

import hashlib
import logging
import random
import tempfile
from pathlib import Path

from PIL import Image

from ic2x.config import Config
from ic2x.utils.ai_client import (
    JudgeCall,
    MultiJudgeCall,
    call_vision_judge,
    call_vision_judge_multi,
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
    """Production rotation check — pick-4 method (see call_rotation_pick4).

    Returns (result_dict, elapsed_seconds, used_network).
    result_dict always has {"upright": bool, "rotate_cw_degrees": int}.
    Fails open: errors return {"upright": True, "rotate_cw_degrees": 0}.
    model_string overrides cfg.rotation_model (used by `autorotate` to compare).
    """
    return call_rotation_pick4(image_path, cfg, model_string=model_string)


def call_rotation_single(image_path: Path, cfg: Config, model_string: str | None = None
                         ) -> tuple[dict, float, bool]:
    """Legacy single-image method: ask the model to estimate the needed rotation.
    Kept for Ollama fallback and A/B comparison; weaker than pick-4 (mental rotation,
    especially 180°, is a known VLM blind spot)."""
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


# ---------------------------------------------------------------------------
# Pick-4: send all 4 rotations, model picks the upright one
# ---------------------------------------------------------------------------

PICK4_PROMPT = """You are shown 4 versions (Image index 0-3) of the SAME photo, each rotated \
differently. Exactly one shows the scene the right way up — as a person standing in it \
would see it.

Judge each version against real-world cues:
- Sky, ceilings, sun and hanging lamps belong at the TOP; floors, tables, ground and \
water at the BOTTOM.
- People and faces are upright (head above shoulders, eyes above mouth).
- Buildings, door frames, trees, poles and bottles stand VERTICALLY.
- Readable text on signs and labels runs horizontally, left-to-right.
- Overhead/table-top shots (meals and plates from above, objects photographed straight \
down) and close-up macros have NO true "up" — every rotation is equally valid.

Return ONLY valid JSON — no markdown:
{"upright_index": <0|1|2|3>, "confident": <true|false>, "reason": "<max 12 words>"}

confident=true ONLY when a clear real-world cue (standing people, buildings, horizon, \
hanging signs, storefront text) fixes the up direction and exactly one version matches \
it. Overhead/table-top/macro shots: ALWAYS confident=false, even if text or object \
placement suggests a natural reading direction. Wrongly rotating a fine photo is worse \
than leaving a tilted one — when in doubt, confident=false. When confident=false the \
photo is left unchanged."""

# Candidate downscale: 4 images per call, so keep each small. Orientation cues
# (horizon, faces, text direction) survive 512px easily; this caps the call at
# roughly the same input tokens as one 1024px single-image call.
PICK4_MAX_PX = 512

# Second-stage guard, only when pick-4 proposes a rotation (rare): real-photo
# errors concentrate on straight-down shots (tabletop meals, first-person shots
# of one's own hands/feet), where pick-4 sometimes takes a confident stance that
# has no basis. One extra 2-image call kills those while keeping genuine fixes.
CONFIRM_PROMPT = """Image index 0 is a photo exactly as it was shot. Image index 1 is the same \
photo rotated {deg} degrees clockwise, proposed as a correction because the original may \
have been stored sideways or upside down.

Decide whether the correction is genuinely needed:
- apply_fix=true ONLY when Image 1 is clearly the natural view and Image 0 is clearly \
wrong: people/buildings/trees stand vertically, sky/ceiling at the top, floor at the \
bottom, readable signs horizontal — in Image 1 but not in Image 0.
- Shots taken looking straight DOWN have no true "up" and must stay as shot → \
apply_fix=false. Telltales: a meal/plates on a table filling the frame, objects seen \
from directly above, the photographer's own hands, feet or shoes in frame, floor/ground \
as the only background.
- If both look plausible, or you are unsure → apply_fix=false.

Return ONLY valid JSON — no markdown:
{{"apply_fix": <true|false>, "reason": "<max 12 words>"}}"""

# PIL's ROTATE_* constants are COUNTER-clockwise; map the CW degrees we speak to them.
_ROT_TRANSPOSE = {90: Image.ROTATE_270, 180: Image.ROTATE_180, 270: Image.ROTATE_90}


def _pick4_shuffle(name: str) -> list[int]:
    """Deterministic per-file presentation order of the 4 CW rotations."""
    order = [0, 90, 180, 270]
    seed = int.from_bytes(hashlib.md5(name.encode()).digest()[:4], "big")
    random.Random(seed).shuffle(order)
    return order


def call_rotation_pick4(image_path: Path, cfg: Config, model_string: str | None = None
                        ) -> tuple[dict, float, bool]:
    """Pick-4 rotation check: build the 4 rotations of image_path, send them in one
    multi-image call (shuffled so index position carries no signal), and apply the
    rotation whose candidate the model picked as upright.

    Same contract as call_rotation_single: returns (result_dict, elapsed, used_network)
    with {"upright": bool, "rotate_cw_degrees": 0|90|180|270}; fails open to upright.
    Cloud models only (multi-image); Ollama falls back to the single-image method.
    """
    from ic2x.utils.image_utils import oriented

    model_str = model_string or cfg.rotation_model
    model, _ = parse_model_effort(model_str)
    if provider_for_model(model) == "ollama":
        return call_rotation_single(image_path, cfg, model_string=model_string)

    max_px = min(PICK4_MAX_PX, cfg.rotation_image_max_px or PICK4_MAX_PX)
    order = _pick4_shuffle(image_path.name)

    try:
        with tempfile.TemporaryDirectory(dir=cfg.work_dir) as td:
            paths: list[Path] = []
            with Image.open(image_path) as im:
                im = oriented(im)
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                w, h = im.size
                if max(w, h) > max_px:
                    s = max_px / max(w, h)
                    im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
                for i, deg in enumerate(order):
                    cand = im if deg == 0 else im.transpose(_ROT_TRANSPOSE[deg])
                    p = Path(td) / f"cand{i}.jpg"
                    cand.save(p, "JPEG", quality=88)
                    paths.append(p)

            parsed, elapsed, ok, used_network = call_vision_judge_multi(
                model_string=model_str,
                ollama_base_url=cfg.ollama_base_url,
                call=MultiJudgeCall(
                    image_paths=paths,
                    prompt=PICK4_PROMPT,
                    max_px=max_px,
                    fail_value=dict(_OK),
                    refused_value=dict(_OK),
                    label="rotation",
                ),
            )

            if not ok or "upright_index" not in parsed:
                if ok:
                    logger.warning("rotation(pick4): unexpected response shape for %s: %s",
                                   image_path.name, str(parsed)[:200])
                return dict(_OK), elapsed, used_network

            try:
                idx = int(parsed.get("upright_index", -1))
            except (TypeError, ValueError):
                idx = -1
            confident = bool(parsed.get("confident", False))
            if idx not in (0, 1, 2, 3) or not confident:
                logger.info("rotation(pick4): keeping %s (index=%s confident=%s reason=%s)",
                            image_path.name, idx, confident,
                            str(parsed.get("reason", ""))[:80])
                return dict(_OK), elapsed, used_network

            degrees = order[idx]
            if degrees == 0:
                return dict(_OK), elapsed, used_network

            # Second stage: confirm the proposed fix against the as-shot version.
            # Fails CLOSED (keep as shot) — a missed fix is recoverable, a wrongly
            # rotated post is not.
            confirm, c_elapsed, c_ok, c_net = call_vision_judge_multi(
                model_string=model_str,
                ollama_base_url=cfg.ollama_base_url,
                call=MultiJudgeCall(
                    image_paths=[paths[order.index(0)], paths[idx]],
                    prompt=CONFIRM_PROMPT.format(deg=degrees),
                    max_px=max_px,
                    fail_value={"apply_fix": False, "reason": "confirm failed"},
                    refused_value={"apply_fix": False, "reason": "confirm refused"},
                    label="rotation_confirm",
                ),
            )
            elapsed += c_elapsed
            used_network = used_network or c_net
            if not c_ok or not bool(confirm.get("apply_fix", False)):
                logger.info("rotation(pick4): confirm kept %s as shot (proposed %d°, %s)",
                            image_path.name, degrees,
                            str(confirm.get("reason", ""))[:80])
                return dict(_OK), elapsed, used_network

            logger.info("rotation(pick4): %s → %d° CW (pick: %s | confirm: %s)",
                        image_path.name, degrees, str(parsed.get("reason", ""))[:60],
                        str(confirm.get("reason", ""))[:60])
            return {"upright": False, "rotate_cw_degrees": degrees}, elapsed, used_network
    except Exception as exc:  # noqa: BLE001 — fail open, never block a post
        logger.warning("rotation(pick4): error for %s: %s", image_path.name, exc)
        return dict(_OK), 0.0, False
