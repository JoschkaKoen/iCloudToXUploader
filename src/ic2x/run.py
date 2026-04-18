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
from ic2x import judge_quality, judge_safety
from ic2x import pull as pull_mod
from ic2x.config import Config, load_config, ensure_dirs
from ic2x.db import DB
from ic2x.utils import ui
from ic2x.utils.ai_client import warmup_ollama, unload_ollama, provider_for_model, parse_model_effort

logger = logging.getLogger("ic2x.run")


def _setup_logging(logs_dir: Path) -> None:
    from rich.logging import RichHandler
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(name)-20s  %(levelname)s  %(message)s")
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            file_handler,
            RichHandler(rich_tracebacks=True, show_path=False),
        ],
    )


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


def run() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    _setup_logging(cfg.logs_dir)

    ui.startup_banner(cfg)

    # Inject proxy into os.environ before any network activity
    if cfg.proxy_http:
        os.environ.setdefault("http_proxy",  cfg.proxy_http)
        os.environ.setdefault("HTTP_PROXY",  cfg.proxy_http)
    if cfg.proxy_https:
        os.environ.setdefault("https_proxy", cfg.proxy_https)
        os.environ.setdefault("HTTPS_PROXY", cfg.proxy_https)

    # Warmup Ollama if either AI model is local — one-time check before the image loop
    _default = os.environ.get("AI_DEFAULT_MODEL", "gemini-2.5-flash")
    _safety_model, _ = parse_model_effort(os.environ.get("SAFETY_MODEL", _default))
    _quality_model, _ = parse_model_effort(os.environ.get("QUALITY_MODEL", _default))
    _ollama_models = {
        m for m in (_safety_model, _quality_model)
        if provider_for_model(m) == "ollama"
    }
    if _ollama_models:
        _ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        for _m in _ollama_models:
            ui.info(f"Warming up Ollama model '{_m}' (may take up to 90s on cold start)…")
            try:
                warmup_ollama(_ollama_base, _m)
                ui.ok(f"Ollama '{_m}' ready")
            except RuntimeError as exc:
                ui.err(str(exc))
                return

    db = DB(cfg.db_path)

    # Startup guard — never silently retry a stuck posting
    stuck = db.get_stuck_posting()
    if stuck:
        ui.err(
            f"Found {len(stuck)} row(s) stuck in 'posting' status. "
            "Manually verify whether the tweet was posted before retrying.\n"
            + "\n".join(f"  sha256={r['sha256']}  file={r['source_filename']}" for r in stuck)
        )
        return

    ui.run_banner()
    files = pull_mod.pull(cfg, db)
    ui.info(f"Pulled {len(files)} file(s) from iCloud")

    queued_count = 0
    rejected_by: dict[str, int] = defaultdict(int)

    for idx, path in enumerate(files, start=1):
        ui.file_header(path.name, idx, len(files))
        sha = ""
        phash = ""

        try:
            # ── [1/6] SHA-256 dedup ────────────────────────────────────────
            ui.stage_banner(1, "DEDUP (SHA-256)")
            sha = dedup.sha256_of(path)
            if db.seen_sha256(sha):
                ui.rejected(path.name, "duplicate", "sha256 already seen")
                _reject(path, cfg, db, sha, "", "duplicate", "sha256", logs_dir=cfg.logs_dir)
                rejected_by["duplicate"] += 1
                continue
            ui.ok(f"new  ({sha[:12]}…)")

            # ── [2/6] Screenshot filter ────────────────────────────────────
            ui.stage_banner(2, "SCREENSHOT CHECK")
            is_ss, ss_reason = filter.is_screenshot(path)
            if is_ss:
                ui.rejected(path.name, "screenshot", ss_reason)
                _reject(path, cfg, db, sha, "", "screenshot", ss_reason, logs_dir=cfg.logs_dir)
                rejected_by["screenshot"] += 1
                continue
            ui.ok("not a screenshot")

            # ── [3/6] pHash dedup ──────────────────────────────────────────
            ui.stage_banner(3, "DEDUP (pHash)")
            phash = dedup.phash_of(path)
            if db.seen_phash_similar(phash, cfg.hamming_threshold):
                ui.rejected(path.name, "duplicate", "perceptual duplicate of posted image")
                _reject(path, cfg, db, sha, phash, "duplicate", "phash", logs_dir=cfg.logs_dir)
                rejected_by["duplicate"] += 1
                continue
            ui.ok("unique")

            db.insert_seen(sha, phash, path.name)

            # ── [4/6] Safety check ─────────────────────────────────────────
            ui.stage_banner(4, "SAFETY CHECK")
            if db.check_daily_ai_limit(cfg.daily_ai_calls):
                ui.warn("Daily AI call limit reached — stopping")
                break
            safety, safety_elapsed = judge_safety.call_safety(path)
            db.increment_ai_calls()
            safety_ms = int(safety_elapsed * 1000)

            if not safety["safe"]:
                stage = "model_refused" if "gemini_refused" in safety["flags"] else "safety"
                ui.rejected(path.name, stage, safety["flags"])
                dest_sub = "model_refused" if stage == "model_refused" else "safety"
                _reject(
                    path, cfg, db, sha, phash, "rejected", safety["flags"],
                    reject_stage=dest_sub,
                    safety_raw=json.dumps(safety),
                    logs_dir=cfg.logs_dir,
                    ai_ms=safety_ms,
                )
                rejected_by[dest_sub] += 1
                continue
            ui.ok(f"safe  ({safety_ms}ms)")
            ui.info(f'Flags       : {", ".join(safety["flags"]) or "none"}')

            # ── [5/6] Quality check ────────────────────────────────────────
            ui.stage_banner(5, "QUALITY CHECK")
            if db.check_daily_ai_limit(cfg.daily_ai_calls):
                ui.warn("Daily AI call limit reached — stopping")
                break
            quality, quality_elapsed = judge_quality.call_quality(path)
            db.increment_ai_calls()
            quality_ms = int(quality_elapsed * 1000)

            if not quality["interesting"]:
                ui.info(f'Description : {quality["description"]}')
                ui.rejected(path.name, "quality", quality["reason"])
                _reject(
                    path, cfg, db, sha, phash, "rejected", quality["reason"],
                    reject_stage="quality",
                    quality_raw=json.dumps(quality),
                    logs_dir=cfg.logs_dir,
                    ai_ms=quality_ms,
                )
                rejected_by["quality"] += 1
                continue
            ui.ok(f"interesting  ({quality_ms}ms)")
            ui.info(f'Caption     : {quality["caption"]}')
            ui.info(f'Description : {quality["description"]}')

            # ── [6/6] Prepare + optional enhance ──────────────────────────
            ui.stage_banner(6, "PREPARE")
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

            # Write quality sidecar JSON
            (cfg.queue_dir / f"{phash}.json").write_text(
                json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            db.set_status(sha, "queued", caption=quality["caption"])
            _log_decision(
                cfg.logs_dir, sha, phash, path.name,
                "queue", "queued", detail=quality["caption"],
            )
            queued_count += 1
            ui.queued(path.name, phash, quality["caption"])

        except Exception as exc:
            logger.error("Unhandled error for %s: %s", path, exc, exc_info=True)
            ui.err(f"Unhandled error for {path.name}: {exc}")
            continue

    ui.run_summary(len(files), queued_count, dict(rejected_by))
    db.close()

    if _ollama_models:
        for _m in _ollama_models:
            ui.info(f"Unloading Ollama model '{_m}'…")
            unload_ollama(_ollama_base, _m)


def _reject(
    path: Path,
    cfg: Config,
    db: DB,
    sha: str,
    phash: str,
    status: str,
    reason,
    reject_stage: str | None = None,
    safety_raw: str | None = None,
    quality_raw: str | None = None,
    logs_dir: Path | None = None,
    ai_ms: int | None = None,
) -> None:
    reject_subdir = reject_stage or (
        "duplicate" if status == "duplicate" else
        "screenshot" if status == "screenshot" else
        "safety"
    )
    dest_dir = cfg.rejected_dir / reject_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(path), dest_dir / path.name)
    except Exception:
        pass  # file may not exist yet (e.g. sha256 hit before download)

    if sha:
        kwargs: dict = {}
        if phash:
            kwargs["phash"] = phash
        if reject_stage:
            kwargs["reject_stage"] = reject_stage
        if reason:
            kwargs["reject_reason"] = json.dumps(reason) if isinstance(reason, list) else str(reason)
        if safety_raw:
            kwargs["safety_raw"] = safety_raw
        if quality_raw:
            kwargs["quality_raw"] = quality_raw

        if db.seen_sha256(sha):
            db.set_status(sha, "rejected", **kwargs)
        else:
            db.insert_seen(sha, phash or "", path.name)
            db.set_status(sha, "rejected", **kwargs)

    if logs_dir:
        _log_decision(
            logs_dir, sha, phash, path.name,
            reject_stage or status, "rejected",
            detail=reason if isinstance(reason, list) else [str(reason)] if reason else [],
            ai_ms=ai_ms,
        )
