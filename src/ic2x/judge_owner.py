"""
Owner-selfie gate: reference-based "is the account owner the main subject?" check.

The owner never wants their own selfies/portraits published (they leaked once —
2026-07-12, a club selfie). Two layers defend against that:
1. The burst judge's composition rule (JUDGE_EXTRA_RULES flag "selfie") — catches
   single-person close-ups on thumbnails, before a winner is even picked.
2. This check on the prepared winner: the owner's reference photo(s) from
   owner_refs/ plus the candidate go into ONE multi-image call, and the model
   decides whether the SAME person is a main subject of the candidate.

Rejection here walks the cycle back to the next scene (reject_stage
"owner_selfie"), so a false positive costs only a different photo being posted.
Fail-open on errors/no-refs — layer 1 still stands; posting is never blocked.
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

logger = logging.getLogger("ic2x.judge_owner")

OWNER_PROMPT = """The first {n_refs} image(s) are REFERENCE photos of the account owner's face.
The LAST image is a CANDIDATE photo being considered for posting.

The owner never publishes photos of themself. Compare faces carefully (features,
hair, build) and decide about the CANDIDATE:
- owner_present: the person from the reference photo(s) is visibly in the candidate.
- main_subject: that person is a MAIN subject — a selfie, a close-up, or their face
  prominent and recognizable. A small, distant, turned-away or blurry background
  appearance is NOT main_subject.
Other people in the candidate (crowds, vendors, performers, friends, strangers) are
irrelevant — only the owner matters here.

Return ONLY valid JSON — no markdown:
{{"owner_present": <bool>, "main_subject": <bool>, "reason": "<max 12 words>"}}"""

_OK: dict = {"owner_main_subject": False, "reason": ""}


def owner_reference_paths(cfg: Config) -> list[Path]:
    """Reference photos of the owner, newest first, capped at 3 (token cost)."""
    d = cfg.owner_refs_dir
    if not d.is_dir():
        return []
    refs = sorted((p for p in d.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic")),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return refs[:3]


def call_owner_check(image_path: Path, cfg: Config, model_string: str | None = None
                     ) -> tuple[dict, float, bool]:
    """Check whether the account owner is a main subject of image_path.

    Returns (result_dict, elapsed_seconds, used_network) with
    {"owner_main_subject": bool, "reason": str}. Fails open (False) on any error,
    missing references, or an ollama-only model — the composition rule in the
    burst judge remains as the first layer.
    """
    # Local face gate first (2026-07-16): embeddings from owner_refs identify the
    # owner more reliably than a VLM, in ~100 ms, without the photo leaving the
    # Mac. Only when the gate can't decide (models missing, unreadable image)
    # does the cloud VLM below run as fallback.
    try:
        from ic2x.face_gate import get_gate
        res = get_gate(cfg).owner_main_subject(image_path)
        if res is not None:
            hit, reason = res
            if hit:
                logger.info("owner-check(face-gate): %s REJECTED (%s)",
                            image_path.name, reason)
            return {"owner_main_subject": hit, "reason": f"face-gate: {reason}"}, 0.0, False
    except Exception as exc:  # noqa: BLE001 — gate is an accelerator, never a blocker
        logger.warning("owner-check: face gate errored (%s) — using VLM", exc)

    refs = owner_reference_paths(cfg)
    if not refs:
        logger.info("owner-check: no reference photos in %s — skipping", cfg.owner_refs_dir)
        return dict(_OK), 0.0, False

    model_str = model_string or cfg.owner_check_model
    model, _ = parse_model_effort(model_str)
    if provider_for_model(model) == "ollama":
        return dict(_OK), 0.0, False

    parsed, elapsed, ok, used_network = call_vision_judge_multi(
        model_string=model_str,
        ollama_base_url=cfg.ollama_base_url,
        call=MultiJudgeCall(
            image_paths=[*refs, image_path],
            prompt=OWNER_PROMPT.format(n_refs=len(refs)),
            max_px=cfg.owner_check_image_max_px,
            fail_value=dict(_OK),
            refused_value=dict(_OK),
            label="owner_check",
        ),
    )
    if not ok or "owner_present" not in parsed or "main_subject" not in parsed:
        if ok:
            logger.warning("owner-check: unexpected response shape for %s: %s",
                           image_path.name, str(parsed)[:200])
        return dict(_OK), elapsed, used_network

    hit = bool(parsed.get("owner_present")) and bool(parsed.get("main_subject"))
    reason = str(parsed.get("reason", ""))[:120]
    if hit:
        logger.info("owner-check: %s REJECTED as owner selfie (%s)", image_path.name, reason)
    return {"owner_main_subject": hit, "reason": reason}, elapsed, used_network
