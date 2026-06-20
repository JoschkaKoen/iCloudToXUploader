"""
The single autonomous loop: iterate iCloud photos newest-ADDED-first, take the
newest unseen burst of similar shots, one cheap-VLM call picks the best + vets
it, and post it. Walks back through history until something is postable.

  ic2x bot            run the loop
  ic2x bot --once     run a single cycle and exit (for testing)

Key reliability rules:
  • Photos come from recently_added() and bursts are assembled straight from the
    LIVE PhotoAsset objects it yields — never re-resolved by id (all.get(id) hangs).
  • asset_index.seen is the single source of truth — a burst is never re-assembled.
  • Atomic per-burst commit: a crash mid-cycle writes nothing and the identical
    burst re-assembles; flush_pending finishes any APPROVED-but-unposted winner.
  • Walk back until postable, bounded by the daily AI-call cap.
  • Poison-burst breaker: a burst whose pre-commit steps keep failing is skipped
    and finally marked seen after BURST_MAX_ATTEMPTS.
  • Screenshots are dropped via the Screenshots smart album; the winner's full-res
    original still passes the fail-closed EXIF gate before posting.
"""

from __future__ import annotations

import hashlib
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

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


def _safe_name(asset_id: str) -> str:
    return hashlib.sha1(asset_id.encode("utf-8")).hexdigest()[:16]


def _hamming(a_hex: str, b_hex: str) -> int:
    return imagehash.hex_to_hash(a_hex) - imagehash.hex_to_hash(b_hex)


def _unlink(p: Path | None) -> None:
    if p is None:
        return
    try:
        p.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def _keep_reviewed(cfg: Config, outcome: str, burst: "Burst", best_index: int, label: str) -> None:
    """Save EVERY thumbnail of a judged burst to reviewed/<date>/, so the grouping
    is visible: a 3-shot burst writes 3 files sharing an `n3__<head>` prefix, with
    the chosen one marked `_WIN`. Browseable during burn-in."""
    if not cfg.keep_reviewed or not burst.members:
        return
    import shutil
    try:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        d = cfg.reviewed_dir / day
        d.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if (c.isalnum() or c in " -") else "_" for c in (label or "")).strip()
        safe = "-".join(safe.split())[:40] or "na"
        n = len(burst.members)
        head = (burst.head or "x")[:8]
        # Common `b<head>_n<N>` prefix so a burst's members sort together; only
        # the chosen member carries the outcome (the siblings were NOT posted).
        for j, m in enumerate(burst.members):
            if j == best_index:
                role = f"WIN-{outcome}" + (f"__{safe}" if safe != "na" else "")
            else:
                role = "sibling"
            shutil.copy(str(m.thumb), str(d / f"b{head}_n{n}__{j}__{role}.jpg"))
    except Exception as exc:  # noqa: BLE001 — observability must never crash a cycle
        logger.warning("keep_reviewed: %s", exc)


def _download(ic: ICloudPhotos, asset: Any, version: str, dest: Path) -> Path | None:
    """Download a rendition from a live asset. Reauth/Throttled propagate to the
    loop; any other failure → None (treated as undecodable, skipped)."""
    try:
        return ic.download(asset, version, dest)
    except (ReauthRequired, PyiCloudThrottled):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("download failed: %s", exc)
        return None


# ── Live, peekable, seen-skipping stream ───────────────────────────────────────

_UNSET = object()


class _Stream:
    """Peekable (meta, live-asset) stream that skips already-decided assets.
    One per cycle; assemble_burst consumes forward through it."""

    def __init__(self, it: Iterator[tuple[Any, Any]], db: DB) -> None:
        self._it = it
        self._db = db
        self._buf: Any = _UNSET

    def _advance(self):
        for meta, asset in self._it:
            if self._db.seen_asset_id(meta.id):
                continue
            return (meta, asset)
        return None

    def peek(self):
        if self._buf is _UNSET:
            self._buf = self._advance()
        return self._buf

    def take(self):
        v = self.peek()
        self._buf = _UNSET
        return v


# ── Burst assembly ─────────────────────────────────────────────────────────────

@dataclass
class BurstMember:
    asset_id: str
    thumb: Path
    phash: str
    asset: Any  # live PhotoAsset — used to download the winner's full-res original


@dataclass
class Burst:
    members: list[BurstMember] = field(default_factory=list)
    aux_seen: list[str] = field(default_factory=list)  # decided-seen, not judged
    head: str | None = None


def assemble_burst(stream: _Stream, cfg: Config, ic: ICloudPhotos, screenshot_ids: set[str]) -> Burst | None:
    """The newest run of consecutive, visually-similar, not-yet-decided still
    images. Screenshots and undecodable assets go to aux_seen rather than judged.
    Returns None only when the stream is exhausted with nothing left."""
    members: list[BurstMember] = []
    aux_seen: list[str] = []
    prev: str | None = None

    while True:
        item = stream.peek()
        if item is None:
            break
        meta, asset = item
        if meta.id in screenshot_ids:
            aux_seen.append(meta.id)
            stream.take()
            continue
        dest = cfg.work_dir / f"thumb_{_safe_name(meta.id)}.jpg"
        thumb = _download(ic, asset, cfg.thumb_version, dest)  # propagates Reauth/Throttled
        if thumb is None:
            aux_seen.append(meta.id)
            stream.take()
            continue
        try:
            ph = dedup.phash_of(thumb)
        except Exception as exc:  # noqa: BLE001
            logger.warning("burst: pHash failed for %s: %s", meta.id, exc)
            aux_seen.append(meta.id)
            stream.take()
            _unlink(thumb)
            continue
        if prev is not None and _hamming(ph, prev) > cfg.burst_hamming_threshold:
            _unlink(thumb)
            break  # next scene → starts the next burst (left peeked)
        if len(members) >= cfg.burst_max_size:
            if prev is not None and _hamming(ph, prev) <= cfg.burst_hamming_threshold:
                aux_seen.append(meta.id)  # consume the near-dup tail past the cap
                stream.take()
                _unlink(thumb)
                continue
            _unlink(thumb)
            break
        members.append(BurstMember(meta.id, dest, ph, asset))
        prev = ph
        stream.take()

    if not members and not aux_seen:
        return None
    head = members[0].asset_id if members else aux_seen[0]
    return Burst(members, aux_seen, head)


# ── Winner preparation ─────────────────────────────────────────────────────────

def _apply_rotation(path: Path, cw_degrees: int) -> None:
    from PIL import Image
    with Image.open(path) as img:
        img.rotate(-cw_degrees, expand=True).save(path, "JPEG", quality=92, exif=b"")
    logger.info("rotation: applied %d° CW to %s", cw_degrees, path.name)


def _prepare_winner(db: DB, cfg: Config, ic: ICloudPhotos, winner: BurstMember, caption: str
                    ) -> tuple[str, dict | None]:
    """Download the winner full-res (from its live asset), run the EXIF screenshot
    net + dedup, prepare, move to approved/. Returns ("transient", info) to retry
    the burst, ("rejected", winner_dict) to walk back, or ("approved", winner_dict)."""
    import shutil
    orig = cfg.work_dir / f"orig_{_safe_name(winner.asset_id)}"
    try:
        ic.download(winner.asset, "original", orig)
    except (ReauthRequired, PyiCloudThrottled):
        raise
    except Exception as exc:  # noqa: BLE001
        return "transient", {"reason": str(exc)}

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
        shutil.move(str(prepared), str(cfg.approved_dir / prepared.name))
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

def run_one_cycle(db: DB, cfg: Config, ic: ICloudPhotos, clients) -> str:
    """Assemble bursts newest-first, judging each, until one posts (or the daily
    AI cap is hit / the library is exhausted). Posts at most one image."""
    screenshot_ids = ic.screenshot_ids()
    stream = _Stream(ic.iter_image_assets(), db)
    thumbs: list[Path] = []
    try:
        while not db.check_daily_ai_limit(cfg.daily_ai_calls):
            burst = assemble_burst(stream, cfg, ic, screenshot_ids)
            if burst is None:
                db.set_state("backward_exhausted", "1")
                return "exhausted"
            thumbs.extend(m.thumb for m in burst.members)
            seen_ids = [m.asset_id for m in burst.members] + burst.aux_seen

            if not burst.members:
                db.commit_burst(seen_ids, None)  # all screenshots/undecodable
                continue

            verdict, _el, used_net = judge_burst([m.thumb for m in burst.members], cfg)
            if used_net:
                db.increment_ai_calls()

            if not verdict.get("safe") or not verdict.get("interesting"):
                db.commit_burst(seen_ids, None)
                reason = verdict.get("reason") or ",".join(verdict.get("flags") or []) or "rejected"
                log_decision(cfg.logs_dir, outcome="rejected", asset_id=burst.head,
                             detail={"n": len(burst.members), "reason": verdict.get("reason"),
                                     "flags": verdict.get("flags")})
                _keep_reviewed(cfg, "unsafe" if not verdict.get("safe") else "boring",
                               burst, verdict["best_index"], reason)
                continue

            winner = burst.members[verdict["best_index"]]
            kind, payload = _prepare_winner(db, cfg, ic, winner, verdict.get("caption", ""))

            if kind == "transient":
                attempts = db.incr_asset_attempts(burst.head)
                if attempts >= cfg.burst_max_attempts:
                    db.commit_burst(seen_ids, None)
                    log_decision(cfg.logs_dir, outcome="error_giveup", asset_id=burst.head,
                                 detail=payload)
                    continue
                log_decision(cfg.logs_dir, outcome="transient_skip", asset_id=burst.head,
                             detail={"attempt": attempts, **(payload or {})})
                continue  # leave unseen; retried next cycle

            if kind == "rejected":
                db.commit_burst(seen_ids, payload)
                log_decision(cfg.logs_dir, outcome="winner_rejected", asset_id=winner.asset_id,
                             detail={"stage": payload.get("reject_stage")})
                _keep_reviewed(cfg, payload.get("reject_stage") or "rejected", burst,
                               verdict["best_index"], payload.get("reject_stage") or "")
                continue

            db.commit_burst(seen_ids, payload)  # APPROVED — atomic, then post
            row = {"sha256": payload["sha256"], "phash": payload["phash"],
                   "caption": payload.get("caption", "")}
            posted = post_one(row, cfg, db, *clients)
            log_decision(cfg.logs_dir, outcome="posted" if posted else "post_failed",
                         asset_id=winner.asset_id,
                         detail={"best_index": verdict["best_index"], "n": len(burst.members),
                                 "caption": payload.get("caption", "")})
            _keep_reviewed(cfg, "posted" if posted else "postfail", burst,
                           verdict["best_index"], payload.get("caption") or "")
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


def bot(once: bool = False) -> None:
    global _stop
    cfg = load_config()
    ensure_dirs(cfg)

    from ic2x.utils.logging_setup import setup_logging
    setup_logging(cfg.logs_dir)
    ui.startup_banner(cfg)

    # ── preflight: fail fast ──
    require_vision_api_credentials(cfg.judge_model, cfg.rotation_model)
    clients = make_clients(cfg)
    ic = ICloudPhotos(cfg)
    try:
        ic.ensure_session()
    except ReauthRequired as exc:
        _notify_reauth(cfg, exc)
        return
    db = DB(cfg.db_path)

    # ── single-cycle test mode: force one cycle (ignore interval + daily cap) ──
    if once:
        try:
            db.reset_stuck_posting()
            flush_pending(db, cfg, clients)
            outcome = run_one_cycle(db, cfg, ic, clients)
            ui.ok(f"one cycle complete: {outcome}")
            if outcome == "posted":
                ui.info(f"chosen image → {cfg.posted_dir}/  (dry_run={cfg.x_dry_run})")
            ui.info(f"decisions logged → {cfg.logs_dir}/")
        except ReauthRequired as exc:
            _notify_reauth(cfg, exc)
        except PyiCloudThrottled as exc:
            ui.err(f"iCloud throttled — try again later: {exc}")
        finally:
            db.close()
        return

    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("_stop", True))
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("_stop", True))
    ui.info(f"bot started — posting every {cfg.post_interval_hours}h "
            f"(dry_run={cfg.x_dry_run}). Ctrl-C to stop.")

    throttles = 0
    while not _stop:
        try:
            db.reset_stuck_posting()
            flush_pending(db, cfg, clients)
            if _due_to_post(db, cfg) and db.count_posts_rolling_24h() < cfg.max_posts_per_day:
                outcome = run_one_cycle(db, cfg, ic, clients)
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
