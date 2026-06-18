"""
The single autonomous loop: sync a local photo index → take the newest unseen
burst → one VLM call picks the best of the burst → post it. Walks back through
history until something is postable; prefers new photos when they arrive.

  ic2x bot   (also bare `ic2x`)

Correctness/reliability rules implemented here:
  • asset_index.seen is the single source of truth — a burst is never re-assembled.
  • Atomic per-burst commit: a crash mid-cycle writes nothing and the identical
    burst re-assembles; flush_pending finishes any APPROVED-but-unposted winner.
  • Walk back until postable, bounded only by the daily AI-call cap.
  • Poison-burst breaker: a burst whose pre-commit steps keep failing is skipped
    (this cycle) and finally marked seen after BURST_MAX_ATTEMPTS.
  • Screenshots are dropped via the index flag; the winner's full-res original
    still passes the fail-closed EXIF gate before posting.
"""

from __future__ import annotations

import hashlib
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import imagehash

from ic2x import dedup, prepare
from ic2x.config import Config, ensure_dirs, load_config
from ic2x.db import DB
from ic2x.filter import is_screenshot
from ic2x.icloud_photos import ICloudPhotos, PyiCloudThrottled, ReauthRequired
from ic2x.judge_burst import judge_burst
from ic2x.post import make_clients, post_one
from ic2x.status import Status
from ic2x.utils import ui
from ic2x.utils.ai_client import require_vision_api_credentials
from ic2x.utils.decision_log import log_decision

logger = logging.getLogger("ic2x.bot")

_stop = False


# ── Asset source (abstracted so the burst logic is unit-testable offline) ──────

class AssetSource(Protocol):
    def download_thumb(self, asset_id: str) -> Path | None:
        """Local path to the asset's thumbnail, or None if unavailable/undecodable.
        Raises ReauthRequired / PyiCloudThrottled — those must reach the loop."""
        ...

    def download_original(self, asset_id: str) -> Path | None:
        """Local path to the full-res original (fresh-resolved), or None."""
        ...


def _safe_name(asset_id: str) -> str:
    return hashlib.sha1(asset_id.encode("utf-8")).hexdigest()[:16]


class ICloudAssetSource:
    """Real source: resolves assets by id and streams renditions to work_dir.
    Caches the live PhotoAsset per cycle so a member's thumb isn't resolved twice."""

    def __init__(self, ic: ICloudPhotos, cfg: Config) -> None:
        self._ic = ic
        self._cfg = cfg
        self._cache: dict[str, Any] = {}

    def download_thumb(self, asset_id: str) -> Path | None:
        asset = self._cache.get(asset_id)
        if asset is None:
            asset = self._ic.get_asset(asset_id)  # may raise Reauth/Throttled
            if asset is None:
                return None
            self._cache[asset_id] = asset
        dest = self._cfg.work_dir / f"thumb_{_safe_name(asset_id)}.jpg"
        try:
            return self._ic.download(asset, self._cfg.thumb_version, dest)
        except (ReauthRequired, PyiCloudThrottled):
            raise
        except Exception as exc:  # noqa: BLE001 — treat as undecodable, skip
            logger.warning("source: thumb download failed for %s: %s", asset_id, exc)
            return None

    def download_original(self, asset_id: str) -> Path | None:
        asset = self._ic.get_asset(asset_id)  # fresh resolve → fresh signed URL (H3)
        if asset is None:
            return None
        dest = self._cfg.work_dir / f"orig_{_safe_name(asset_id)}"
        return self._ic.download(asset, "original", dest)  # raises on failure


# ── Burst assembly ─────────────────────────────────────────────────────────────

@dataclass
class BurstMember:
    asset_id: str
    thumb: Path
    phash: str  # hex pHash of the thumbnail (for grouping only)


@dataclass
class Burst:
    members: list[BurstMember] = field(default_factory=list)
    aux_seen: list[str] = field(default_factory=list)  # decided-seen, not judged
    head: str | None = None


def _hamming(a_hex: str, b_hex: str) -> int:
    return imagehash.hex_to_hash(a_hex) - imagehash.hex_to_hash(b_hex)


def _unlink(p: Path | None) -> None:
    if p is None:
        return
    try:
        p.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def find_next_burst(db: DB, source: AssetSource, cfg: Config, exclude: set[str]) -> Burst | None:
    """The newest run of consecutive, visually-similar, not-yet-decided still
    images. Screenshots and undecodable assets are collected into aux_seen (to be
    marked seen) rather than judged. Returns None when nothing is left to do."""
    window = db.next_unseen_assets(cfg.burst_max_size * 4 + len(exclude))
    window = [r for r in window if r["asset_id"] not in exclude]
    if not window:
        return None

    burst = Burst(head=window[0]["asset_id"])
    prev: str | None = None
    for row in window:
        aid = row["asset_id"]
        if row["is_screenshot"]:
            burst.aux_seen.append(aid)
            continue
        thumb = source.download_thumb(aid)  # propagates Reauth/Throttled
        if thumb is None:
            burst.aux_seen.append(aid)  # undecodable / unavailable → seen, not a boundary
            continue
        try:
            ph = dedup.phash_of(thumb)
        except Exception as exc:  # noqa: BLE001
            logger.warning("burst: pHash failed for %s: %s", aid, exc)
            burst.aux_seen.append(aid)
            _unlink(thumb)
            continue
        if prev is not None and _hamming(ph, prev) > cfg.burst_hamming_threshold:
            _unlink(thumb)
            break  # next scene → starts the next burst; leave it unseen
        if len(burst.members) >= cfg.burst_max_size:
            if prev is not None and _hamming(ph, prev) <= cfg.burst_hamming_threshold:
                burst.aux_seen.append(aid)  # consume the near-dup tail past the cap (M6)
                _unlink(thumb)
                continue
            _unlink(thumb)
            break
        burst.members.append(BurstMember(aid, thumb, ph))
        prev = ph

    if burst.members:
        burst.head = burst.members[0].asset_id
    return burst


# ── Winner preparation ─────────────────────────────────────────────────────────

def _apply_rotation(path: Path, cw_degrees: int) -> None:
    from PIL import Image
    with Image.open(path) as img:
        img.rotate(-cw_degrees, expand=True).save(path, "JPEG", quality=92, exif=b"")
    logger.info("rotation: applied %d° CW to %s", cw_degrees, path.name)


def _prepare_winner(
    db: DB, cfg: Config, source: AssetSource, winner: BurstMember, caption: str,
) -> tuple[str, dict | None]:
    """Download the winner full-res, run the EXIF screenshot net + dedup, prepare,
    and move to approved/. Returns one of:
      ("transient", exc)        — retry the whole burst (network/IO failure)
      ("rejected", winner_dict) — decided-out (screenshot/duplicate); walk back
      ("approved", winner_dict) — ready to post
    """
    orig: Path | None = None
    try:
        orig = source.download_original(winner.asset_id)
    except (ReauthRequired, PyiCloudThrottled):
        raise
    except Exception as exc:  # noqa: BLE001
        return "transient", {"reason": str(exc)}
    if orig is None:
        return "transient", {"reason": "original unavailable"}

    try:
        sha = dedup.sha256_of(orig)
        ph = dedup.phash_of(orig)
        is_ss, ss_reason = is_screenshot(orig)
        if is_ss:
            return "rejected", {
                "asset_id": winner.asset_id, "sha256": sha, "phash": ph,
                "status": Status.REJECTED, "reject_stage": "screenshot",
                "reject_reason": ss_reason, "caption": "",
            }
        if db.seen_sha256(sha) or db.seen_phash_similar(ph, cfg.hamming_threshold):
            return "rejected", {
                "asset_id": winner.asset_id, "sha256": sha, "phash": ph,
                "status": Status.REJECTED, "reject_stage": "duplicate",
                "reject_reason": "sha/phash near-duplicate of a kept image", "caption": "",
            }
        prepared = prepare.prepare(orig, cfg.queue_dir, ph)
        if cfg.rotation_enabled:
            try:
                from ic2x import judge_rotation
                rot, _el, used = judge_rotation.call_rotation(prepared, cfg)
                if used:
                    db.increment_ai_calls()
                if not rot["upright"] and rot["rotate_cw_degrees"]:
                    _apply_rotation(prepared, rot["rotate_cw_degrees"])
            except Exception as exc:  # noqa: BLE001 — optional; never block the post
                logger.warning("rotation: skipped (%s)", exc)
        import shutil
        dest = cfg.approved_dir / prepared.name
        shutil.move(str(prepared), str(dest))
        return "approved", {
            "asset_id": winner.asset_id, "sha256": sha, "phash": ph,
            "filename": winner.asset_id, "status": Status.APPROVED, "caption": caption,
        }
    except (ReauthRequired, PyiCloudThrottled):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("winner prepare failed for %s: %s", winner.asset_id, exc, exc_info=True)
        return "transient", {"reason": str(exc)}
    finally:
        _unlink(orig)


# ── Cycle ──────────────────────────────────────────────────────────────────────

def run_one_cycle(db: DB, cfg: Config, source: AssetSource, clients) -> str:
    """Assemble bursts newest-first, judging each, until one posts (or the daily
    AI cap is hit / the library is exhausted). Posts at most one image."""
    skip: set[str] = set()
    thumbs: list[Path] = []
    try:
        while not db.check_daily_ai_limit(cfg.daily_ai_calls):
            burst = find_next_burst(db, source, cfg, skip)
            if burst is None:
                db.set_state("backward_exhausted", "1")
                return "exhausted"
            db.set_state("backward_exhausted", "0")
            thumbs.extend(m.thumb for m in burst.members)
            seen_ids = [m.asset_id for m in burst.members] + burst.aux_seen

            if not burst.members:
                db.commit_burst(seen_ids, None)  # window was all screenshots/undecodable
                continue

            verdict, _el, used_net = judge_burst([m.thumb for m in burst.members], cfg)
            if used_net:
                db.increment_ai_calls()

            if not verdict.get("safe") or not verdict.get("interesting"):
                db.commit_burst(seen_ids, None)
                log_decision(cfg.logs_dir, outcome="rejected", asset_id=burst.head,
                             detail={"n": len(burst.members), "reason": verdict.get("reason"),
                                     "flags": verdict.get("flags")})
                continue

            winner = burst.members[verdict["best_index"]]
            kind, payload = _prepare_winner(db, cfg, source, winner, verdict.get("caption", ""))

            if kind == "transient":
                attempts = db.incr_asset_attempts(burst.head)
                if attempts >= cfg.burst_max_attempts:
                    db.commit_burst(seen_ids, None)  # give up; unblock walk-back
                    log_decision(cfg.logs_dir, outcome="error_giveup", asset_id=burst.head,
                                 detail=payload)
                    continue
                skip.add(burst.head)  # retry whole burst a future wakeup; move on now
                log_decision(cfg.logs_dir, outcome="transient_skip", asset_id=burst.head,
                             detail={"attempt": attempts, **(payload or {})})
                continue

            if kind == "rejected":
                db.commit_burst(seen_ids, payload)
                log_decision(cfg.logs_dir, outcome="winner_rejected", asset_id=winner.asset_id,
                             detail={"stage": payload.get("reject_stage")})
                continue

            # approved → commit the whole burst atomically, then post the winner
            db.commit_burst(seen_ids, payload)
            row = {"sha256": payload["sha256"], "phash": payload["phash"],
                   "caption": payload.get("caption", "")}
            posted = post_one(row, cfg, db, *clients)
            log_decision(cfg.logs_dir, outcome="posted" if posted else "post_failed",
                         asset_id=winner.asset_id,
                         detail={"best_index": verdict["best_index"], "n": len(burst.members),
                                 "caption": payload.get("caption", "")})
            return "posted" if posted else "post_failed"
        return "ai_cap"
    finally:
        for t in thumbs:
            _unlink(t)


def flush_pending(db: DB, cfg: Config, clients) -> bool:
    """Post at most ONE leftover APPROVED image (crash recovery). Respects the
    rolling-24h cap so a post-outage replay can't spray the account."""
    approved = db.get_approved()
    if not approved:
        return False
    if db.count_posts_rolling_24h() >= cfg.max_posts_per_day:
        return False
    return post_one(approved[0], cfg, db, *clients)


# ── Index sync ─────────────────────────────────────────────────────────────────

_SYNC_STOP_AFTER_KNOWN = 50  # consecutive already-indexed assets before assuming caught up


def sync_index(db: DB, ic: ICloudPhotos, cfg: Config) -> int:
    """Incrementally index still images, newest-capture-first. Stops after a run
    of already-known assets (first run indexes the whole library). Returns #new."""
    new = 0
    known_streak = 0
    for meta, _asset in ic.iter_image_assets():  # metadata only
        if db.asset_indexed(meta.id):
            known_streak += 1
            if known_streak >= _SYNC_STOP_AFTER_KNOWN:
                break
            continue
        known_streak = 0
        if db.upsert_asset(meta.id, meta.created, meta.filename,
                           is_live=meta.is_live, width=meta.width, height=meta.height):
            new += 1
    if new:
        logger.info("sync_index: %d new assets (%d total)", new, db.asset_index_count())
    return new


def refresh_screenshots_if_due(db: DB, ic: ICloudPhotos, cfg: Config) -> None:
    raw = db.get_state("ss_refresh_counter")
    counter = int(raw) if raw and raw.isdigit() else 0
    if counter > 0:
        db.set_state("ss_refresh_counter", str(counter - 1))
        return
    ids = ic.screenshot_ids()
    if ids:
        n = db.mark_screenshots(list(ids))
        logger.info("screenshots: flagged %d known assets", n)
    db.set_state("ss_refresh_counter", str(cfg.screenshot_album_refresh_cycles))


# ── Loop ───────────────────────────────────────────────────────────────────────

def _notify_reauth(cfg: Config, exc: Exception) -> None:
    ui.err(f"iCloud re-auth required: {exc}  →  run `ic2x login`")
    if cfg.reauth_notify_cmd:
        import subprocess
        try:
            subprocess.run(cfg.reauth_notify_cmd, shell=True, timeout=30, check=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("reauth_notify_cmd failed: %s", e)


def _sleep_interruptible(seconds: float) -> None:
    end = time.monotonic() + seconds
    while not _stop and time.monotonic() < end:
        time.sleep(1)


def _due_to_post(db: DB, cfg: Config) -> bool:
    last = db.get_last_posted_at()
    if last is None:
        return True
    return datetime.now(timezone.utc) - last >= timedelta(hours=cfg.post_interval_hours)


def bot() -> None:
    global _stop
    cfg = load_config()
    ensure_dirs(cfg)

    from ic2x.utils.logging_setup import setup_logging
    setup_logging(cfg.logs_dir)
    ui.startup_banner(cfg)

    # ── preflight: fail fast before the loop ──
    require_vision_api_credentials(cfg.judge_model, cfg.rotation_model)
    clients = make_clients(cfg)
    ic = ICloudPhotos(cfg)
    try:
        ic.ensure_session()
    except ReauthRequired as exc:
        _notify_reauth(cfg, exc)
        return
    db = DB(cfg.db_path)

    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("_stop", True))
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("_stop", True))
    ui.info(f"bot started — posting every {cfg.post_interval_hours}h "
            f"(dry_run={cfg.x_dry_run}). Ctrl-C to stop.")

    throttles = 0
    while not _stop:
        try:
            db.reset_stuck_posting()
            refresh_screenshots_if_due(db, ic, cfg)
            flush_pending(db, cfg, clients)

            if _due_to_post(db, cfg) and db.count_posts_rolling_24h() < cfg.max_posts_per_day:
                sync_index(db, ic, cfg)
                outcome = run_one_cycle(db, cfg, ICloudAssetSource(ic, cfg), clients)
                logger.info("cycle: %s", outcome)
            throttles = 0
        except ReauthRequired as exc:
            _notify_reauth(cfg, exc)
            break
        except PyiCloudThrottled as exc:
            throttles += 1
            backoff = min(cfg.daemon_check_interval * (2 ** throttles), 3600)
            logger.warning("throttled (%d) — backing off %ds: %s", throttles, backoff, exc)
            _sleep_interruptible(backoff)
            continue
        except Exception as exc:  # noqa: BLE001 — one bad cycle must not kill the bot
            logger.error("cycle error: %s", exc, exc_info=True)

        _sleep_interruptible(cfg.daemon_check_interval)

    db.close()
    ui.info("bot stopped.")
