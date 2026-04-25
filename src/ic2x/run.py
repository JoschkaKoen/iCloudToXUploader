"""
Orchestrator for `ic2x run`.

Pull photos from iCloud → filter → dedup → safety check → quality check
→ prepare (rotate + JPEG + EXIF strip) → optional InstructIR enhancement
→ drop in queue/ for manual review.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pillow_heif  # noqa: F401
pillow_heif.register_heif_opener()

from ic2x import dedup, enhance, filter, prepare
from ic2x import judge_rotation, judge_safety_quality
from ic2x import pull as pull_mod
from ic2x.group import cluster_by_phash
from ic2x.config import Config, load_config, ensure_dirs
from ic2x.db import DB
from ic2x.status import Status
from ic2x.utils import ui
from ic2x.utils.logging_setup import setup_logging
from ic2x.utils.ai_client import warmup_ollama, unload_ollama, provider_for_model, parse_model_effort  # noqa: F401

logger = logging.getLogger("ic2x.run")

_TOTAL_STAGES = 6  # SHA-256 dedup, screenshot, pHash dedup, safety+quality, prepare, rotation


def _log_decision(
    logs_dir: Path,
    sha256: str,
    phash: str,
    filename: str,
    stage: str,
    outcome: str,
    detail=None,
    ai_ms: int | None = None,
) -> None:
    record: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256,
        "phash": phash,
        "filename": filename,
        "stage": stage,
        "outcome": outcome,
    }
    if detail is not None:
        record["detail"] = detail
    if ai_ms is not None:
        record["ai_ms"] = ai_ms

    log_file = logs_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run(
    recent_override: int = 0,
    auto_unstick: bool = False,
    show_banner: bool = True,
) -> tuple[int, int, int]:
    """Run the full pull → filter → judge → prepare pipeline.

    Returns (new_pulled, queued_count, approved_count).
    show_banner=False suppresses the startup config table (used by daemon).
    """
    cfg = load_config()
    ensure_dirs(cfg)
    setup_logging(cfg.logs_dir)

    if show_banner:
        ui.startup_banner(cfg)

    # Inject proxy into os.environ before any network activity. This must
    # happen before openai/httpx/tweepy import their underlying TLS clients.
    if cfg.proxy_http:
        os.environ.setdefault("http_proxy",  cfg.proxy_http)
        os.environ.setdefault("HTTP_PROXY",  cfg.proxy_http)
    if cfg.proxy_https:
        os.environ.setdefault("https_proxy", cfg.proxy_https)
        os.environ.setdefault("HTTPS_PROXY", cfg.proxy_https)

    # Warmup Ollama if either AI model is local — one-time check before the image loop
    _judge_m, _    = parse_model_effort(cfg.judge_model)
    _rotation_m, _ = parse_model_effort(cfg.rotation_model)
    _ollama_models = {
        m for m in (_judge_m, _rotation_m) if provider_for_model(m) == "ollama"
    }
    if _ollama_models:
        for _m in _ollama_models:
            ui.info(f"Warming up Ollama model '{_m}' (may take up to 90s on cold start)…")
            try:
                warmup_ollama(cfg.ollama_base_url, _m)
                ui.ok(f"Ollama '{_m}' ready")
            except RuntimeError as exc:
                ui.err(str(exc))
                return 0, 0, 0

    db = DB(cfg.db_path)

    # Startup guard — never silently retry a stuck posting
    stuck = db.get_stuck_posting()
    if stuck:
        if auto_unstick:
            logger.warning(
                "run: auto-resetting %d stuck 'posting' row(s) (daemon mode)", len(stuck)
            )
            db.reset_stuck_posting()
        else:
            ui.err(
                f"Found {len(stuck)} row(s) stuck in 'posting' status. "
                "Manually verify whether the tweet was posted before retrying.\n"
                + "\n".join(f"  sha256={r['sha256']}  file={r['source_filename']}" for r in stuck)
            )
            db.close()
            return 0, 0, 0

    if show_banner:
        ui.run_banner()
    files = pull_mod.pull(cfg, db, recent_override=recent_override)
    new_pulled = len(files)
    ui.info(f"Pulled {new_pulled} file(s) from iCloud")

    queued_count = 0
    approved_count = 0
    rejected_by: dict[str, int] = defaultdict(int)

    for idx, path in enumerate(files, start=1):
        ui.file_header(path.name, idx, len(files))
        sha = ""
        phash = ""

        try:
            # ── [1/6] SHA-256 dedup ────────────────────────────────────────
            ui.stage_banner(1, _TOTAL_STAGES, "DEDUP (SHA-256)")
            sha = dedup.sha256_of(path)
            _existing = db.get_image_by_sha(sha)
            if _existing is not None and _existing["status"] != Status.SEEN.value:
                # Fully processed or rejected before — true duplicate
                ui.rejected(path.name, "duplicate", "sha256 already seen")
                _reject(path, cfg, db, sha, "", "duplicate", "sha256")
                rejected_by["duplicate"] += 1
                continue
            # _existing is None (new image) or status=='seen' (crashed mid-pipeline → resume)
            if _existing is not None:
                logger.info("run: resuming crashed pipeline run for %s", path.name)
            ui.ok(f"new  ({sha[:12]}…)")

            # ── [2/6] Screenshot filter ────────────────────────────────────
            ui.stage_banner(2, _TOTAL_STAGES, "SCREENSHOT CHECK")
            is_ss, ss_reason = filter.is_screenshot(path)
            if is_ss:
                ui.rejected(path.name, "screenshot", ss_reason)
                _reject(path, cfg, db, sha, "", "screenshot", ss_reason)
                rejected_by["screenshot"] += 1
                continue
            ui.ok("not a screenshot")

            # ── [3/6] pHash dedup ──────────────────────────────────────────
            ui.stage_banner(3, _TOTAL_STAGES, "DEDUP (pHash)")
            phash = dedup.phash_of(path)
            if db.seen_phash_similar(phash, cfg.hamming_threshold):
                ui.rejected(path.name, "duplicate", "perceptual near-duplicate of already-processed image")
                _reject(path, cfg, db, sha, phash, "duplicate", "phash")
                rejected_by["duplicate"] += 1
                continue
            ui.ok("unique")

            if _existing is None:
                db.insert_seen(sha, phash, path.name)

            # ── [4/6] Safety + quality check ──────────────────────────────
            ui.stage_banner(4, _TOTAL_STAGES, "SAFETY + QUALITY CHECK")
            if db.check_daily_ai_limit(cfg.daily_ai_calls):
                ui.warn("Daily AI call limit reached — stopping")
                break
            result, sq_elapsed = judge_safety_quality.call_safety_quality(path, cfg)
            db.increment_ai_calls()
            sq_ms = int(sq_elapsed * 1000)

            if not result["safe"]:
                stage = "model_refused" if "gemini_refused" in result["flags"] else "safety"
                ui.rejected(path.name, stage, result["flags"])
                dest_sub = "model_refused" if stage == "model_refused" else "safety"
                _reject(
                    path, cfg, db, sha, phash, dest_sub, result["flags"],
                    safety_raw=json.dumps({"safe": result["safe"], "flags": result["flags"]}),
                    ai_ms=sq_ms,
                )
                rejected_by[dest_sub] += 1
                continue
            ui.ok(f"safe  ({sq_ms}ms)")
            ui.info(f'Flags       : {", ".join(result["flags"]) or "none"}')

            if not result["interesting"]:
                ui.info(f'Description : {result["description"]}')
                ui.info(f'Reason      : {result["reason"]}')
                ui.rejected(path.name, "quality", result["reason"])
                _reject(
                    path, cfg, db, sha, phash, "quality", result["reason"],
                    quality_raw=json.dumps({k: result[k] for k in ("interesting", "description", "caption", "reason")}),
                    ai_ms=sq_ms,
                )
                rejected_by["quality"] += 1
                continue
            ui.ok(f"interesting  ({sq_ms}ms)")
            ui.info(f'Caption     : {result["caption"]}')
            ui.info(f'Description : {result["description"]}')

            # ── [5/6] Prepare + optional enhance ──────────────────────────
            ui.stage_banner(5, _TOTAL_STAGES, "PREPARE")
            prepared = prepare.prepare(path, cfg.queue_dir, phash)

            if cfg.enhance_enabled:
                enhanced = enhance.enhance_image(
                    prepared,
                    ir_dir=cfg.enhance_instructir_dir,
                    prompt=cfg.enhance_prompt,
                    enabled=True,
                )
                if enhanced != prepared:
                    prepared = enhanced
                    ui.ok(f"{prepared.name}  [enhanced via InstructIR]")
                else:
                    ui.ok(f"{prepared.name}  [enhance attempted but skipped]")
            else:
                ui.ok(f"{prepared.name}  [enhance: disabled]")

            # ── [6/6] Visual rotation check ───────────────────────────────
            ui.stage_banner(6, _TOTAL_STAGES, "ROTATION CHECK")
            if db.check_daily_ai_limit(cfg.daily_ai_calls):
                ui.warn("Daily AI call limit reached — skipping rotation check")
            else:
                rotation, rotation_elapsed = judge_rotation.call_rotation(prepared, cfg)
                db.increment_ai_calls()
                rotation_ms = int(rotation_elapsed * 1000)
                degrees = rotation["rotate_cw_degrees"]
                if not rotation["upright"] and degrees != 0:
                    _apply_rotation(prepared, degrees)
                    ui.ok(f"rotated {degrees}°  ({rotation_ms}ms)")
                else:
                    ui.ok(f"upright  ({rotation_ms}ms)")

            # Write quality sidecar JSON to queue_dir first
            sidecar_path = cfg.queue_dir / f"{phash}.json"
            sidecar_data = {k: result[k] for k in ("interesting", "description", "caption", "reason")}
            sidecar_path.write_text(
                json.dumps(sidecar_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            if cfg.auto_approve:
                # Move image + sidecar directly to approved/ — skip manual review
                dest_img  = cfg.approved_dir / prepared.name
                dest_json = cfg.approved_dir / f"{phash}.json"
                shutil.move(str(prepared), dest_img)
                shutil.move(str(sidecar_path), dest_json)
                db.set_status(sha, Status.APPROVED, caption=result["caption"])
                _log_decision(
                    cfg.logs_dir, sha, phash, path.name,
                    "queue", "auto-approved", detail=result["caption"],
                )
                approved_count += 1
                ui.info(f"[AUTO-APPROVE] → approved/{phash[:12]}…")
            else:
                db.set_status(sha, Status.QUEUED, caption=result["caption"])
                _log_decision(
                    cfg.logs_dir, sha, phash, path.name,
                    "queue", "queued", detail=result["caption"],
                )
                queued_count += 1
                ui.queued(path.name, phash, result["caption"])

        except Exception as exc:
            logger.error("Unhandled error for %s: %s", path, exc, exc_info=True)
            ui.err(f"Unhandled error for {path.name}: {exc}")
            continue

    ui.run_summary(new_pulled, queued_count + approved_count, dict(rejected_by))

    if cfg.group_hamming_threshold > 0:
        pending = db.get_pending()
        if len(pending) >= 2:
            groups = cluster_by_phash(pending, cfg.group_hamming_threshold)
            for group in groups:
                if len(group) > 1:
                    names = " + ".join((r["phash"] or "")[:16] + ".jpg" for r in group)
                    ui.info(f"Will post together ({len(group)} images): {names}")

    db.close()

    if _ollama_models:
        for _m in _ollama_models:
            ui.info(f"Unloading Ollama model '{_m}'…")
            unload_ollama(cfg.ollama_base_url, _m)

    return new_pulled, queued_count, approved_count


def _reject(
    path: Path,
    cfg: Config,
    db: DB,
    sha: str,
    phash: str,
    reject_stage: str,                 # also the rejected/<stage>/ subfolder name
    reason,                            # str or list[str] — flagged reasons
    *,
    safety_raw: str | None = None,
    quality_raw: str | None = None,
    ai_ms: int | None = None,
) -> None:
    """Move a rejected image to rejected/<stage>/, mark the row REJECTED,
    and append a decision record to logs/YYYY-MM-DD.jsonl.
    """
    dest_dir = cfg.rejected_dir / reject_stage
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(str(path), dest_dir / path.name)
        # Original stays in inbox/ — icloudpd will find it and skip re-downloading next run
    except Exception:
        pass  # file may not exist yet (e.g. sha256 hit before download)

    if sha:
        kwargs: dict = {"reject_stage": reject_stage}
        if phash:
            kwargs["phash"] = phash
        if reason:
            kwargs["reject_reason"] = json.dumps(reason) if isinstance(reason, list) else str(reason)
        if safety_raw:
            kwargs["safety_raw"] = safety_raw
        if quality_raw:
            kwargs["quality_raw"] = quality_raw

        if not db.seen_sha256(sha):
            db.insert_seen(sha, phash or "", path.name)
        db.set_status(sha, Status.REJECTED, **kwargs)

    _log_decision(
        cfg.logs_dir, sha, phash, path.name,
        reject_stage, "rejected",
        detail=reason if isinstance(reason, list) else [str(reason)] if reason else [],
        ai_ms=ai_ms,
    )


def _apply_rotation(path: Path, cw_degrees: int) -> None:
    """Rotate a JPEG clockwise by cw_degrees and overwrite it in place."""
    from PIL import Image
    with Image.open(path) as img:
        # PIL rotate() is counter-clockwise; negate for clockwise rotation
        rotated = img.rotate(-cw_degrees, expand=True)
        rotated.save(path, "JPEG", quality=92, exif=b"")
    logger.info("rotation: applied %d° CW to %s", cw_degrees, path.name)
