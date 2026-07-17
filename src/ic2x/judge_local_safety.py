"""
Local (Ollama) safety pre-filter — Tier 1 of judge localization (2026-07-16).

Runs BEFORE any thumbnail is uploaded to a cloud judge: a local VLM screens the
whole burst against the same hard safety rules the cloud judge enforces
(students, selfies, close-up faces in nightlife, private documents…). A local
"unsafe" verdict rejects the burst ON-DEVICE — those photos never leave the Mac.

Fail-THROUGH by design: prefilter disabled, model missing, Ollama down, low RAM,
bad JSON, any exception → ({"safe": True}, ran=False) and the normal cloud
pipeline proceeds (which still screens safety). The pre-filter can only ADD
privacy; it can never weaken safety or block posting.

RAM guard: skips local inference when free+reclaimable memory is below
cfg.local_prefilter_min_free_gb (the 30b model needs ~20 GB resident; running
it into a full machine is how you crash a MacBook — user requirement).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ic2x.config import Config
from ic2x.judge_safety_quality import SAFETY_BLOCK
from ic2x.utils.ai_client import call_ollama_chat
from ic2x.utils.image_utils import encode_image_b64
from ic2x.utils.response_parsing import parse_json_safe

logger = logging.getLogger("ic2x.local_safety")

_PROMPT_TMPL = """You are a strict PRIVACY AND SAFETY screen for photos about to be posted to a
public X (twitter) account. You see N images, in order ("Image index i", i from 0).
Decide whether ANY of them must not be posted.

{safety_block}

{extra_rules}

Judge ONLY safety/privacy — ignore whether the photos are interesting.
Return ONLY valid JSON — no markdown:
{{"safe": <bool — false when ANY image violates a rule>, "flags": ["<rule tags>"], "reason": "<max 12 words>"}}"""


def ram_free_gb() -> float:
    """Kernel free-memory level (kern.memorystatus_level, elastic-aware percent)
    expressed as GB on this machine. vm_stat free+inactive undercounts badly on
    macOS (drops to ~3 GB the moment Ollama loads while the kernel still reports
    >80%); the memorystatus level is the kernel's own availability estimate. It
    dips transiently DURING inference, so callers must only gate BEFORE issuing
    a call, never abort mid-flight."""
    import os
    total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    pct = int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"],
                             capture_output=True, text=True, timeout=10).stdout.strip())
    return pct / 100 * total_gb


def shadow_judge_burst(thumbs: list[Path], cfg: Config) -> tuple[dict, bool]:
    """Tier-2 pilot: run the FULL production burst-judge prompt on the local
    model and return its verdict for agreement logging. NEVER affects the
    posting decision — the caller only compares and logs. Same fail-through
    contract as prefilter_burst."""
    ok: dict = {}
    if not getattr(cfg, "local_judge_shadow", False) or not thumbs:
        return ok, False
    try:
        free = ram_free_gb()
        floor = float(getattr(cfg, "local_prefilter_min_free_gb", 10) or 10)
        if free < floor:
            return ok, False
        from ic2x.judge_burst import burst_prompt
        imgs = [encode_image_b64(p, max_px=cfg.judge_image_max_px) for p in thumbs]
        raw = call_ollama_chat(
            cfg.ollama_base_url,
            getattr(cfg, "local_judge_model", "qwen3-vl:30b"),
            burst_prompt(cfg),
            imgs,
            think=False,
        )
        parsed = parse_json_safe(raw)
        if not isinstance(parsed, dict) or "safe" not in parsed:
            return ok, False
        return parsed, True
    except Exception as exc:  # noqa: BLE001 — shadow must never affect the cycle
        logger.warning("local-shadow: error (%s)", str(exc)[:120])
        return ok, False


def prefilter_burst(thumbs: list[Path], cfg: Config) -> tuple[dict, bool]:
    """Screen burst thumbnails locally. Returns (verdict, ran).

    verdict always has {"safe": bool, "flags": list, "reason": str}; ran=True only
    when a real local inference produced the verdict (ran=False → fail-through,
    caller must proceed with the cloud pipeline unchanged)."""
    ok: dict = {"safe": True, "flags": [], "reason": ""}
    if not getattr(cfg, "local_prefilter_enabled", False) or not thumbs:
        return ok, False
    try:
        free = ram_free_gb()
        floor = float(getattr(cfg, "local_prefilter_min_free_gb", 10) or 10)
        if free < floor:
            logger.warning("local-prefilter: skipped — free RAM %.1f GB < %.0f GB floor",
                           free, floor)
            return ok, False

        prompt = _PROMPT_TMPL.format(
            safety_block=SAFETY_BLOCK,
            extra_rules=(cfg.judge_extra_rules or "").strip(),
        )
        imgs = [encode_image_b64(p, max_px=cfg.judge_image_max_px) for p in thumbs]
        raw = call_ollama_chat(
            cfg.ollama_base_url,
            getattr(cfg, "local_prefilter_model", "qwen3-vl:8b"),
            prompt,
            imgs,
            think=False,
        )
        parsed = parse_json_safe(raw)
        if not isinstance(parsed, dict) or "safe" not in parsed:
            logger.warning("local-prefilter: bad response shape — falling through (%s)",
                           str(raw)[:120])
            return ok, False
        verdict = {
            "safe": bool(parsed.get("safe", True)),
            "flags": [str(f) for f in (parsed.get("flags") or [])][:6],
            "reason": str(parsed.get("reason", ""))[:120],
        }
        return verdict, True
    except Exception as exc:  # noqa: BLE001 — privacy add-on must never block the bot
        logger.warning("local-prefilter: error — falling through to cloud (%s)",
                       str(exc)[:120])
        return ok, False
