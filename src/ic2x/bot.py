"""
The single autonomous loop: iterate iCloud photos newest-ADDED-first, take the
newest unseen burst of similar shots, one cheap-VLM call picks the best + vets
it, and post it. Walks back through history until something is postable.

  ic2x bot            run the loop
  ic2x bot --once     run a single cycle and exit (for testing)
  ic2x bot --test     fast dry-run soak into test_run/<timestamp>/ (for testing)

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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import imagehash

from ic2x import dedup, prepare
from ic2x.config import Config, _PROJECT_ROOT, ensure_dirs, load_config, require_credentials
from ic2x.db import DB
from ic2x.filter import is_screenshot
from ic2x.icloud_photos import ICloudPhotos, PyiCloudThrottled, ReauthRequired
from ic2x.judge_burst import judge_burst
from ic2x.post import make_clients, post_one
from ic2x.status import Status
from ic2x.utils import ui
from ic2x.utils.ai_client import (
    get_run_usage, require_vision_api_credentials, reset_run_usage,
)
from ic2x.utils.cost_report import (
    compute_cost, format_total_cost_line, format_usage_log, pricing_currency,
)
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
    """Persist a judged burst so the grouping is browsable under reviewed/<date>/:
    a MULTI-shot burst becomes its OWN folder (every member inside, the chosen
    shot suffixed `_WIN`); a SINGLE-shot burst is a loose `…_WIN.jpg` file in the
    day folder. Folder/file names carry the time + outcome + reason/caption, so
    the results read at a glance. Browseable during burn-in."""
    if not cfg.keep_reviewed or not burst.members:
        return
    import shutil
    try:
        now = datetime.now(timezone.utc)
        day = cfg.reviewed_dir / now.strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        lab = "".join(c if (c.isalnum() or c in " -") else "_" for c in (label or "")).strip()
        lab = "-".join(lab.split())[:30]
        tag = "__".join(p for p in (now.strftime("%H%M%S"), outcome, lab) if p)

        def _stem(m, j: int) -> str:
            base = (getattr(m, "filename", "") or m.asset_id or f"img{j}").rsplit(".", 1)[0]
            return ("".join(c if (c.isalnum() or c in "-_") else "_" for c in base)[:40]) or f"img{j}"

        if len(burst.members) == 1:  # singleton → loose file in the day folder
            m = burst.members[0]
            shutil.copy(str(m.thumb), str(day / f"{tag}__{_stem(m, 0)}_WIN.jpg"))
        else:                         # group → its own folder, winner marked _WIN
            grp = day / tag
            grp.mkdir(parents=True, exist_ok=True)
            for j, m in enumerate(burst.members):
                win = "_WIN" if j == best_index else ""
                shutil.copy(str(m.thumb), str(grp / f"{j:02d}_{_stem(m, j)}{win}.jpg"))
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


# ── Live, peekable, seen-skipping, prefetching stream ───────────────────────────

@dataclass
class _Ready:
    """A burst candidate whose thumbnail has been prefetched + hashed."""
    meta: Any
    asset: Any
    thumb: Path | None       # downloaded thumbnail, or None (screenshot / failed)
    phash: str | None        # perceptual hash, or None
    is_screenshot: bool
    dl_failed: bool = False  # thumb download failed (transient) — retry, don't mark seen


class _Stream:
    """Peekable, seen-skipping stream of prefetched burst candidates (one per
    cycle). Each batch's thumbnails are downloaded + hashed CONCURRENTLY in a
    bounded pool but surfaced in capture order, so assemble_burst's sequential
    logic is unchanged while the per-download network round-trip is hidden.
    `concurrency` is also the batch size; keep it modest (~4–8) so parallel GETs
    don't trip iCloud throttling. A worker's Reauth/Throttled propagates out."""

    def __init__(self, it: Iterator[tuple[Any, Any]], db: DB, cfg: Config,
                 ic: ICloudPhotos, screenshot_ids: set[str], *, concurrency: int = 6) -> None:
        self._it = it
        self._db = db
        self._cfg = cfg
        self._ic = ic
        self._ss = screenshot_ids
        self._n = max(1, concurrency)
        self._buf: deque[_Ready] = deque()
        self._exhausted = False
        self._scanned = 0  # already-reviewed photos skipped this cycle (for progress)

    def _raw_next(self):
        for meta, asset in self._it:
            if self._db.seen_asset_id(meta.id):
                self._scanned += 1
                if self._scanned % 200 == 0:
                    ui.info(f"   … {self._scanned} recent photos already reviewed — "
                            "still scanning for a new one")
                continue
            return (meta, asset)
        return None

    def _prepare(self, meta: Any, asset: Any) -> _Ready:
        """Download + hash one candidate (runs in a worker thread). Screenshots
        are flagged without downloading. Reauth/Throttled propagate."""
        if meta.id in self._ss:
            return _Ready(meta, asset, None, None, True)
        dest = self._cfg.work_dir / f"thumb_{_safe_name(meta.id)}.jpg"
        thumb = _download(self._ic, asset, self._cfg.thumb_version, dest)
        if thumb is None:
            # Transient download failure — flag it so assembly retries the photo
            # on a later cycle instead of permanently marking it seen.
            return _Ready(meta, asset, None, None, False, dl_failed=True)
        ph: str | None = None
        try:
            ph = dedup.phash_of(thumb)
        except Exception as exc:  # noqa: BLE001
            logger.warning("burst: pHash failed for %s: %s", meta.id, exc)
            _unlink(thumb)
            thumb = None
        return _Ready(meta, asset, thumb, ph, False)

    def _fill(self) -> None:
        raw: list[tuple[Any, Any]] = []
        for _ in range(self._n):
            nxt = self._raw_next()
            if nxt is None:
                self._exhausted = True
                break
            raw.append(nxt)
        if not raw:
            return
        if len(raw) == 1:
            self._buf.append(self._prepare(*raw[0]))
            return
        # Download + hash the batch concurrently; ex.map keeps capture order and
        # re-raises a worker's Reauth/Throttled in order.
        with ThreadPoolExecutor(max_workers=min(self._n, len(raw))) as ex:
            for ready in ex.map(lambda ma: self._prepare(ma[0], ma[1]), raw):
                self._buf.append(ready)

    def peek(self) -> "_Ready | None":
        if not self._buf and not self._exhausted:
            self._fill()
        return self._buf[0] if self._buf else None

    def take(self) -> "_Ready | None":
        v = self.peek()
        if self._buf:
            self._buf.popleft()
        return v

    def close(self) -> None:
        """Unlink prefetched-but-unconsumed thumbnails so work/ doesn't leak."""
        while self._buf:
            r = self._buf.popleft()
            if r.thumb is not None:
                _unlink(r.thumb)


# ── Burst assembly ─────────────────────────────────────────────────────────────

@dataclass
class BurstMember:
    asset_id: str
    thumb: Path
    phash: str
    asset: Any  # live PhotoAsset — used to download the winner's full-res original
    filename: str = ""  # original iCloud filename (e.g. IMG_7795.HEIC), for readable archives


@dataclass
class Burst:
    members: list[BurstMember] = field(default_factory=list)
    aux_seen: list[str] = field(default_factory=list)  # decided-seen, not judged
    head: str | None = None


def _ai_same_scene(candidate_thumb: Path, head_thumb: Path, cfg: Config, db: "DB") -> bool:
    """Cheap VLM: is `candidate_thumb` the SAME scene/object as the burst's head
    shot (just a different orientation / crop / angle)? Used to MERGE pHash-split
    variants — pHash can't tell portrait-vs-landscape of one scene from a new
    scene. Fails CLOSED (→ "new scene") so any error just keeps plain pHash behavior."""
    try:
        from ic2x import judge_scene_dedup
        dup, used = judge_scene_dedup.call_scene_dedup(
            candidate_thumb, [head_thumb], cfg, model_string=cfg.scene_dedup_model)
        if used:
            db.increment_support_calls()
        return dup is not None
    except Exception as exc:  # noqa: BLE001 — grouping aid, never crash assembly
        logger.warning("scene_group: check failed (%s) — treating as new scene", exc)
        return False


def assemble_burst(stream: _Stream, cfg: Config, ic: ICloudPhotos, screenshot_ids: set[str],
                   db: "DB | None" = None) -> Burst | None:
    """The newest run of consecutive, not-yet-decided still images forming ONE
    scene, from the stream's prefetched candidates. Grouping is pHash by default;
    when `cfg.scene_group_enabled` and `db` are set, a pHash boundary is double-
    checked by a cheap VLM ("same scene shot differently?") so portrait/landscape/
    angle variants stay in one burst. Screenshots/undecodable go to aux_seen.
    Returns None only when the stream is exhausted. (`ic`/`screenshot_ids` retained
    for back-compat; the stream owns download + screenshot-skip.)"""
    members: list[BurstMember] = []
    aux_seen: list[str] = []
    prev: str | None = None
    grouping = bool(getattr(cfg, "scene_group_enabled", False)) and db is not None

    while True:
        r = stream.peek()
        if r is None:
            break
        if r.dl_failed:
            stream.take()
            # Transient download failure — leave UNSEEN so the photo is retried
            # next cycle ("go back in time only when no newest unposted image was
            # detected/downloaded"); the attempts breaker bounds poison assets.
            max_att = getattr(cfg, "burst_max_attempts", 3)
            if db is not None and db.incr_asset_attempts(r.meta.id) >= max_att:
                aux_seen.append(r.meta.id)
                logger.warning("burst: %s failed download %d× — giving up (seen)",
                               r.meta.id, cfg.burst_max_attempts)
            continue
        if r.is_screenshot or r.thumb is None or r.phash is None:
            aux_seen.append(r.meta.id)  # screenshot / undecodable / hash-failed
            stream.take()
            continue
        ph = r.phash
        same_scene = prev is None or _hamming(ph, prev) <= cfg.burst_hamming_threshold
        if not same_scene and grouping and members:
            # pHash says NEW scene, but it can't see a crop/rotation as the same
            # scene — ask the cheap VLM before splitting the burst.
            same_scene = _ai_same_scene(r.thumb, members[0].thumb, cfg, db)
            if same_scene:
                ui.info(f"scene-group: merged {getattr(r.meta, 'filename', None) or 'a shot'}"
                        " into this scene (same scene, reframed/rotated)")
        if not same_scene:
            break  # genuinely a new scene → starts the next burst (left peeked)
        if len(members) >= cfg.burst_max_size:
            aux_seen.append(r.meta.id)   # scene continues past the cap → consume as seen
            stream.take()
            _unlink(r.thumb)
            continue
        members.append(BurstMember(r.meta.id, r.thumb, ph, r.asset,
                                   getattr(r.meta, "filename", "")))
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


def _apply_polish(path: Path, cfg: Config) -> None:
    """Bake the free, local adaptive polish into the prepared JPEG in place. The
    polish is itself fail-open and adaptive (a well-exposed photo comes out nearly
    unchanged), so this only re-encodes when polish is enabled."""
    from PIL import Image

    from ic2x import polish as polish_mod
    with Image.open(path) as img:
        out = polish_mod.polish(img, cfg)
        out.save(path, "JPEG", quality=92, exif=b"")
    logger.info("polish: applied %s to %s", getattr(cfg, "polish_intensity", "natural"), path.name)


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
        # Where + when this was shot, from the original's GPS/EXIF — read HERE, before
        # prepare()/rotation strip EXIF from the posted JPEG. Both feed the caption pass
        # below and the 📍 line (we share the city as text, never the coordinates).
        from ic2x import geo
        place = when = None
        try:
            place, when = geo.place_and_time(orig, cfg)
        except Exception as exc:  # noqa: BLE001 — optional; never block the post
            logger.warning("location: skipped (%s)", exc)

        prepared = prepare.prepare(orig, cfg.queue_dir, ph)
        # Owner-selfie gate: the owner's own face as main subject never gets posted.
        # Runs FIRST among the winner passes — a rejection walks back before any
        # rotation/enhance/caption spend. Fail-open (layer 1 = judge "selfie" flag).
        if getattr(cfg, "owner_check_enabled", False):
            try:
                from ic2x import judge_owner
                ow, _oel, oused = judge_owner.call_owner_check(prepared, cfg)
                if oused:
                    db.increment_ai_calls()
                if ow["owner_main_subject"]:
                    ui.warn(f"   ↳ skipped — owner selfie ({ow.get('reason', '')})")
                    _unlink(prepared)
                    return "rejected", {
                        "asset_id": winner.asset_id, "sha256": sha, "phash": ph,
                        "status": Status.REJECTED, "reject_stage": "owner_selfie",
                        "reject_reason": ow.get("reason", "owner is main subject"),
                        "caption": "",
                    }
                ui.info("   ↳ owner check: not a selfie ✓")
            except (ReauthRequired, PyiCloudThrottled):
                raise
            except Exception as exc:  # noqa: BLE001 — optional; never block the post
                logger.warning("owner-check: skipped (%s)", exc)
        if cfg.rotation_enabled:
            try:
                from ic2x import judge_rotation
                rot, _el, used = judge_rotation.call_rotation(prepared, cfg)
                if used:
                    db.increment_ai_calls()
                if not rot["upright"] and rot["rotate_cw_degrees"]:
                    _apply_rotation(prepared, rot["rotate_cw_degrees"])
                    ui.info(f"   ↳ rotation: was stored {rot['rotate_cw_degrees']}° off — "
                            f"fixed ({rot.get('reason', 'AI pick-4')})")
                else:
                    ui.info("   ↳ rotation check: upright ✓")
            except Exception as exc:  # noqa: BLE001 — optional; never block the post
                logger.warning("rotation: skipped (%s)", exc)
        # Free local adaptive polish (CPU-only). Runs BEFORE the paid color enhance, so
        # it can stand alone (color enhance off) or feed into it. Fail-open by design.
        if getattr(cfg, "polish_enabled", False):
            try:
                _apply_polish(prepared, cfg)
            except Exception as exc:  # noqa: BLE001 — optional; never block the post
                logger.warning("polish: skipped (%s)", exc)
        # Auto color enhancement (Aliyun VIAPI EnhanceImageColor) on the posted image.
        # Fail-open: any failure logs and posts the un-enhanced JPEG. The flat per-call
        # cost flows into the daily cost tracker so `ic2x cost` shows total spend.
        if getattr(cfg, "color_enhance_enabled", False):
            try:
                from ic2x.utils import aliyun_viapi
                if aliyun_viapi.enhance_post_image(
                        prepared, prepared, mode=cfg.color_enhance_mode,
                        max_edge=cfg.color_enhance_max_edge):
                    # first N calls/month are free (Aliyun); only charge beyond that.
                    month_n = db.increment_color_calls()
                    free = getattr(cfg, "color_enhance_free_quota", 100)
                    cost = 0.0 if month_n <= free else getattr(cfg, "color_enhance_cost_rmb", 0.02)
                    if cost:
                        db.add_run_cost(cost)            # persist to the day's total
                    from ic2x.utils.ai_client import add_run_flat_cost
                    add_run_flat_cost(                   # run total + live cost line
                        cost, f"color-enhance {cfg.color_enhance_mode} [{month_n}/{free}/mo]")
            except Exception as exc:  # noqa: BLE001 — enhancement must never block a post
                logger.warning("color enhance: skipped (%s)", exc)
        # Final caption: a dedicated pass on the prepared image that KNOWS the place +
        # local time, so it ties what's visible to real local knowledge instead of
        # generic filler. Best-effort — keep the judge's caption if it fails.
        if getattr(cfg, "caption_pass_enabled", True):
            try:
                from ic2x import caption as caption_mod
                new_cap, used = caption_mod.generate_caption(prepared, place, when, cfg)
                for _ in range(int(used)):   # best-of-N: one increment per network call
                    db.increment_ai_calls()
                if new_cap:
                    caption = new_cap
            except Exception as exc:  # noqa: BLE001 — never block a post on captioning
                logger.warning("caption: pass skipped (%s)", exc)

        cfg.approved_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(prepared), str(cfg.approved_dir / prepared.name))
        # Cap to X's 280-char limit by WEIGHTED length (emoji/CJK count as 2), trimming
        # the caption — never the 📍 line. Plain len() would let a caption with a couple
        # of emoji + the location line slip over 280 weighted and 403 at post time.
        from ic2x.utils.tweet_text import truncate_weighted, weighted_len
        location = geo.format_location_line(place, when)
        caption = (caption or "").strip()
        if location:
            room = max(0, 280 - weighted_len(location) - 1)  # -1 for the "\n"
            body = truncate_weighted(caption, room).rstrip()
            # Don't emit a leading blank line when the caption is empty — post just
            # the 📍 line in that case.
            final_caption = f"{body}\n{location}" if body else location
        else:
            final_caption = truncate_weighted(caption, 280)
        return "approved", {
            "asset_id": winner.asset_id, "sha256": sha, "phash": ph,
            "filename": winner.asset_id, "status": Status.APPROVED, "caption": final_caption,
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
    reset_run_usage()  # isolate this cycle's AI usage for cost accounting
    ui.info("🔍 scanning iCloud for the newest photo worth posting …")
    # Cached Screenshots-album membership (refreshed every N cycles) so we don't
    # re-iterate the whole smart album from iCloud every cycle. getattr keeps the
    # lightweight test fakes (which only define screenshot_ids) working.
    _ss_cached = getattr(ic, "screenshot_ids_cached", None)
    screenshot_ids = (_ss_cached(cfg.screenshot_album_refresh_cycles)
                      if _ss_cached else ic.screenshot_ids())
    # Chronological catalog walk when the client supports it (real ICloudPhotos);
    # plain iteration for lightweight test fakes.
    _chrono = getattr(ic, "iter_image_assets_chrono", None)
    stream = _Stream(_chrono(db) if _chrono else ic.iter_image_assets(),
                     db, cfg, ic, screenshot_ids,
                     concurrency=cfg.prefetch_concurrency)
    thumbs: list[Path] = []
    def _capped() -> bool:
        """The walk-back stops on either budget. DAILY_AI_CALLS bounds DECISION calls
        (judge, owner, rotation, caption) — the ones that actually produce a post.
        Cheap scene-grouping calls fire at every pHash boundary and are bounded
        separately by the much larger DAILY_SUPPORT_CALLS guard: billing them to the
        same meter starved the search (2026-07-17: 172 of 195 calls were grouping, so
        only 19 bursts were ever judged and the walk-back stopped for the day)."""
        if db.check_daily_ai_limit(cfg.daily_ai_calls):
            return True
        if db.check_daily_support_limit(getattr(cfg, "daily_support_calls", 1500)):
            logger.warning("assembly-call guard hit (%d/day) — grouping walked a long "
                           "way without finding a postable burst",
                           getattr(cfg, "daily_support_calls", 1500))
            return True
        return False

    try:
        while not _capped():
            # Honour a stop request between bursts. A cycle can walk back through
            # hundreds of photos, so without this check SIGTERM/Ctrl-C waited for the
            # whole cycle: systemd's default 90s stop timeout expired and it SIGKILLed
            # the bot mid-scan (2026-08-04). Stopping here is always safe — a burst is
            # committed atomically before its post, and flush_pending finishes any
            # APPROVED-but-unposted winner on the next start.
            if _stop:
                return "interrupted"
            burst = assemble_burst(stream, cfg, ic, screenshot_ids, db)
            if burst is None:
                db.set_state("backward_exhausted", "1")
                return "exhausted"
            thumbs.extend(m.thumb for m in burst.members)
            seen_ids = [m.asset_id for m in burst.members] + burst.aux_seen

            if not burst.members:
                db.commit_burst(seen_ids, None)  # all screenshots/undecodable
                continue

            # Local safety pre-filter (Tier 1 localization): screens the burst
            # ON-DEVICE before any thumbnail is uploaded. Unsafe → reject here,
            # photos never leave the Mac. Fail-through: cloud judge still screens.
            if getattr(cfg, "local_prefilter_enabled", False):
                from ic2x import judge_local_safety
                lv, ran = judge_local_safety.prefilter_burst(
                    [m.thumb for m in burst.members], cfg)
                if ran and not lv.get("safe", True):
                    db.commit_burst(seen_ids, None)
                    lflags = ",".join(lv.get("flags") or []) or "local_unsafe"
                    log_decision(cfg.logs_dir, outcome="rejected_local", asset_id=burst.head,
                                 detail={"n": len(burst.members), "flags": lv.get("flags"),
                                         "reason": lv.get("reason")})
                    _keep_reviewed(cfg, "unsafe", burst, 0, f"local:{lflags}")
                    ui.warn(f"   ↳ skipped — UNSAFE, caught on-device [{lflags}] "
                            "(never uploaded)")
                    continue

            n = len(burst.members)
            ui.info(f"judging burst — {n} shot{'s' if n != 1 else ''} …")
            verdict, _el, used_net = judge_burst([m.thumb for m in burst.members], cfg)
            if used_net:
                db.increment_ai_calls()
            # Tier-2 pilot: local judge runs in SHADOW — agreement is logged,
            # the cloud verdict always decides. Flip LOCAL_JUDGE_* config to
            # promote once a week of shadow data looks clean.
            if getattr(cfg, "local_judge_shadow", False):
                from ic2x import judge_local_safety
                sv, s_ran = judge_local_safety.shadow_judge_burst(
                    [m.thumb for m in burst.members], cfg)
                if s_ran:
                    a_safe = bool(sv.get("safe", True)) == bool(verdict.get("safe"))
                    a_int = bool(sv.get("interesting", False)) == bool(verdict.get("interesting"))
                    tag = "agrees" if (a_safe and a_int) else "DISAGREES"
                    ui.info(f"   ↳ local shadow judge {tag} "
                            f"(safe {sv.get('safe')}/{verdict.get('safe')}, "
                            f"interesting {sv.get('interesting')}/{verdict.get('interesting')})")
                    log_decision(cfg.logs_dir, outcome="local_shadow", asset_id=burst.head,
                                 detail={"agree_safe": a_safe, "agree_interesting": a_int,
                                         "local": {k: sv.get(k) for k in
                                                   ("safe", "interesting", "best_index", "flags")},
                                         "cloud": {k: verdict.get(k) for k in
                                                   ("safe", "interesting", "best_index", "flags")}})
            shows = (verdict.get("shows") or "").strip()
            if shows:
                ui.info(f"   ↳ shows: {shows}")

            # A judge that ERRORED (no client, bad response shape) has said nothing
            # about these photos. It fails closed so nothing unsafe can slip through,
            # but committing that as a rejection would permanently burn the burst on
            # what is usually a transient blip — observed 2026-07-17, two bursts lost
            # to error:no_client and never reconsidered. Route it through the same
            # attempts breaker as a transient download failure instead. A genuine
            # `model_refused` is NOT an error and still rejects below.
            _errs = [str(f) for f in (verdict.get("flags") or []) if str(f).startswith("error:")]
            if _errs:
                attempts = db.incr_asset_attempts(burst.head)
                if attempts >= cfg.burst_max_attempts:
                    db.commit_burst(seen_ids, None)
                    log_decision(cfg.logs_dir, outcome="error_giveup", asset_id=burst.head,
                                 detail={"n": len(burst.members), "stage": "judge",
                                         "flags": _errs})
                    ui.warn(f"   ↳ judge kept failing [{','.join(_errs)}] after "
                            f"{attempts} tries — giving up on this burst")
                    continue
                log_decision(cfg.logs_dir, outcome="transient_skip", asset_id=burst.head,
                             detail={"attempt": attempts, "stage": "judge", "flags": _errs})
                ui.warn(f"   ↳ judge unavailable [{','.join(_errs)}] — leaving this burst "
                        "unseen for the next cycle")
                continue

            # Posting bar: the judge scores the chosen shot 0-10; anything under
            # cfg.quality_min_score walks back even when interesting=true. Missing/
            # invalid score → 0 (fail-closed), matching judge_burst's validation.
            try:
                quality = int(verdict.get("quality", 0))
            except (TypeError, ValueError):
                quality = 0
            min_q = getattr(cfg, "quality_min_score", 7)
            if not verdict.get("safe") or not verdict.get("interesting") or quality < min_q:
                db.commit_burst(seen_ids, None)
                reason = verdict.get("reason") or ",".join(verdict.get("flags") or []) or "rejected"
                log_decision(cfg.logs_dir, outcome="rejected", asset_id=burst.head,
                             detail={"n": len(burst.members), "reason": verdict.get("reason"),
                                     "flags": verdict.get("flags"), "quality": quality})
                outcome_word = ("unsafe" if not verdict.get("safe")
                                else "boring" if not verdict.get("interesting")
                                else f"lowq{quality}")
                _keep_reviewed(cfg, outcome_word, burst, verdict["best_index"], reason)
                if not verdict.get("safe"):
                    ui.warn(f"   ↳ skipped — UNSAFE [{','.join(verdict.get('flags') or []) or 'flagged'}]")
                elif not verdict.get("interesting"):
                    ui.info(f"   ↳ skipped — not interesting (quality {quality}/10): "
                            f"{reason} (walking back)")
                else:
                    ui.info(f"   ↳ skipped — quality {quality}/10 below the bar of {min_q}: "
                            f"{reason} (walking back)")
                continue

            winner = burst.members[verdict["best_index"]]

            # Same-scene gate (cheap VLM): reject re-posting the SAME scene/object
            # as a recent post — orientation/crop/angle differences that pHash
            # misses. On the thumb, BEFORE the full-res download. Fails open.
            if cfg.scene_dedup_enabled:
                recent = [(ph, cfg.scene_thumbs_dir / f"{ph}.jpg")
                          for ph in db.recent_posted_phashes(cfg.scene_dedup_recent_n)]
                recent = [(ph, p) for ph, p in recent if p.exists()]
                if recent:
                    from ic2x import judge_scene_dedup
                    dup, used = judge_scene_dedup.call_scene_dedup(
                        winner.thumb, [p for _, p in recent], cfg)
                    if used:
                        db.increment_ai_calls()
                    if dup is not None:
                        db.commit_burst(seen_ids, None)
                        log_decision(cfg.logs_dir, outcome="duplicate_scene",
                                     asset_id=winner.asset_id,
                                     detail={"n": len(burst.members), "same_as": recent[dup][0]})
                        _keep_reviewed(cfg, "dup_scene", burst, verdict["best_index"],
                                       f"same scene as {recent[dup][0]}")
                        ui.info("   ↳ skipped — same scene as a recent post (walking back)")
                        continue

            ui.info(f"   ↳ best of {len(burst.members)} is #{verdict['best_index']} "
                    f"(quality {quality}/10) — downloading "
                    f"full-res{' + enhancing' if getattr(cfg, 'color_enhance_enabled', False) else ''} …")
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
                ui.info(f"   ↳ skipped — winner {payload.get('reject_stage') or 'rejected'} (walking back)")
                continue

            db.commit_burst(seen_ids, payload)  # APPROVED — atomic, then post
            _cap = (payload.get("caption") or "").replace("\n", "  ")
            if _cap:
                ui.info(f"      “{_cap}”")
            ui.ok("   ↳ posting to X …")
            row = {"sha256": payload["sha256"], "phash": payload["phash"],
                   "caption": payload.get("caption", "")}
            posted = post_one(row, cfg, db, *clients)
            log_decision(cfg.logs_dir, outcome="posted" if posted else "post_failed",
                         asset_id=winner.asset_id,
                         detail={"best_index": verdict["best_index"], "n": len(burst.members),
                                 "quality": quality, "caption": payload.get("caption", "")})
            _keep_reviewed(cfg, "posted" if posted else "postfail", burst,
                           verdict["best_index"], payload.get("caption") or "")
            return "posted" if posted else "post_failed"
        return "ai_cap"
    finally:
        # persist this cycle's token spend to the day's total (the live per-call cost
        # lines are shown by the meter; this just keeps `ic2x cost` accurate).
        usage = get_run_usage()
        cost, _bd = compute_cost(usage)
        if cost:
            db.add_run_cost(cost)
            logger.info("cost: %s",
                        format_usage_log(usage, cost_line=format_total_cost_line(cost)))
        for t in thumbs:
            _unlink(t)
        stream.close()  # drop any prefetched-but-unconsumed thumbnails


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


def _sleep_interruptible(seconds: float, heartbeat=None, heartbeat_every: float = 300.0) -> None:
    """Sleep up to `seconds`, waking immediately on Ctrl-C. With `heartbeat`, call it
    now and then every `heartbeat_every`s, so a long wait keeps printing a liveness line
    instead of looking frozen."""
    if heartbeat:
        heartbeat()
    end = time.monotonic() + seconds
    next_beat = time.monotonic() + heartbeat_every
    while not _stop and time.monotonic() < end:
        time.sleep(1)
        if heartbeat and time.monotonic() >= next_beat:
            heartbeat()
            next_beat += heartbeat_every


def _due_to_post(db: DB, cfg: Config) -> bool:
    last = db.get_last_posted_at()
    if last is None:
        return True
    return datetime.now(timezone.utc) - last >= timedelta(hours=cfg.post_interval_hours)


def _waiting_status(db: DB, cfg: Config) -> None:
    """One-line heartbeat while the loop waits, so a normal run always visibly shows it's
    alive AND what it's waiting on — daily cap, the next-post countdown, or simply no new
    photo yet. An otherwise-silent 5h gap looks frozen; this never lets it."""
    posted_today = db.count_posts_rolling_24h()
    if posted_today >= cfg.max_posts_per_day:
        ui.info(f"⏳ idle — daily cap reached ({posted_today}/{cfg.max_posts_per_day}); "
                "waiting for a 24h slot to free. Ctrl-C to stop.")
        return
    last = db.get_last_posted_at()
    if last is not None:
        secs = (last + timedelta(hours=cfg.post_interval_hours)
                - datetime.now(timezone.utc)).total_seconds()
        if secs > 0:
            ui.info(f"⏳ idle — next post in ~{int(secs // 3600)}h {int(secs % 3600 // 60):02d}m "
                    f"({posted_today}/{cfg.max_posts_per_day} today). Ctrl-C to stop.")
            return
    # Due to post, but nothing new turned up this scan — keep polling iCloud.
    ui.info(f"⏳ waiting for a new photo to post — re-checking iCloud every "
            f"{max(1, cfg.daemon_check_interval // 60)} min. Ctrl-C to stop.")


def _tidy_empty_dirs(cfg: Config) -> None:
    """Remove content dirs that ended the run empty — work/queue/approved hold files
    only transiently, so a run leaves no empty folders behind. rmdir() only removes a
    dir when it's actually empty, so anything still holding files is left alone."""
    for d in (cfg.work_dir, cfg.queue_dir, cfg.approved_dir,
              cfg.posted_dir, cfg.scene_thumbs_dir, cfg.reviewed_dir):
        try:
            d.rmdir()
        except OSError:
            pass


def bot(once: bool = False, test: bool = False, test_cycles: int = 20,
        post: bool = False) -> None:
    global _stop
    # A dead HTTPS connection without a timeout blocks an SSL read FOREVER and
    # silently freezes the whole loop mid-scan (observed 2026-07-14: 0% CPU,
    # stuck in poll() after "scanning iCloud…"). pyicloud issues requests with
    # no timeout, so give every socket a generous default — a stalled read then
    # raises within 90s and flows into the existing transient-retry handling.
    import socket
    socket.setdefaulttimeout(90)
    cfg = load_config()

    # ── fast test-run mode: no waits + bypass the post interval / daily cap ──
    # By default --test does NOT post to X: it redirects ALL state/output under a fresh
    # test_run/<timestamp>/ folder — re-runnable and non-destructive. Adding --post makes
    # it a LIVE speed-run (posts to X for REAL) and keeps the REAL state.db / paths so
    # posts are tracked, preventing duplicate re-posts across runs. Test-mode posting is
    # controlled by --post (NOT X_DRY_RUN). MUST run before ensure_dirs/setup_logging/
    # make_clients/DB. icloud_cookie_dir (the 2FA session) is never redirected.
    if test or cfg.test_mode:
        cfg.test_mode = True
        cfg.x_dry_run = not post        # default: no X posting (dry); --post → live posts
        cfg.keep_reviewed = True        # always write the reviewed/ grouping+dedup artifacts
        if cfg.x_dry_run:               # dry → isolated throwaway state; live → real state
            root = _PROJECT_ROOT / "test_run" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
            cfg.db_path = root / "state.db"
            cfg.work_dir = root / "work"
            cfg.queue_dir = root / "queue"
            cfg.approved_dir = root / "approved"
            cfg.posted_dir = root / "posted"
            cfg.scene_thumbs_dir = root / "scene_thumbs"
            cfg.reviewed_dir = root / "reviewed"
            cfg.logs_dir = root / "logs"

    ensure_dirs(cfg)

    from ic2x.utils.logging_setup import setup_logging
    setup_logging(cfg.logs_dir)
    ui.startup_banner(cfg)
    if cfg.test_mode:
        if cfg.x_dry_run:
            ui.warn(f"🧪 TEST MODE (dry) · no waits · output → "
                    f"{cfg.logs_dir.parent.relative_to(_PROJECT_ROOT)}/ · real state untouched; "
                    "iCloud + AI calls are REAL")
        else:
            ui.warn("🧪 TEST MODE (LIVE POSTS) · no waits · posting to X for REAL on the real "
                    "state.db · iCloud + AI + posts all REAL")

    # ── preflight: fail fast ──
    # iCloud is always needed to fetch photos; X creds only for REAL posts (a dry-run
    # burn-in needs no Twitter keys). Validated here so the failure is one clear line.
    require_credentials(cfg, icloud=True, x=not cfg.x_dry_run)
    require_vision_api_credentials(cfg.judge_model, cfg.rotation_model)
    if cfg.color_enhance_enabled:
        from ic2x.utils.aliyun_viapi import ViapiError, check_credentials
        try:
            check_credentials()
        except ViapiError as exc:
            ui.err(str(exc))
            return
    clients = make_clients(cfg)
    ic = ICloudPhotos(cfg)
    # An expired 2FA session must NOT kill a continuous run. `--once` is a one-shot
    # command, so it still fails fast with one clear line; the long-running bot instead
    # starts in a reauth-pending state and keeps polling, so a later `ic2x login` (fresh
    # cookies on disk) revives it unattended. Before this, the errors>=6 re-exec below
    # dropped the loop's own reauth resilience into this exit path and the bot died for
    # good — observed 2026-07-20: 15 days down after a routine ~30-day session expiry.
    reauth_pending = False
    try:
        ic.ensure_session()
    except ReauthRequired as exc:
        _notify_reauth(cfg, exc)
        if once or cfg.test_mode:
            return
        reauth_pending = True
    db = DB(cfg.db_path)

    # startup reconciliation, at the start of EVERY bot run: sync the DB with what's
    # live on X — re-queue posts the owner deleted (reads X via the Bearer token). A
    # --test run has an empty isolated DB, so reconcile is a no-op there.
    if cfg.reconcile_on_startup:
        if cfg.twitter_bearer_token:
            from ic2x.reconcile import make_x_lookup, reconcile_with_x
            reconcile_with_x(db, cfg, make_x_lookup(cfg))
        else:
            ui.warn("reconcile — set TWITTER_BEARER_TOKEN to enable startup DB↔X sync")

    # live per-call cost meter: every priced AI call (and the flat color-enhance cost)
    # prints its cost with the running run-total right-aligned.
    from ic2x.utils.ai_client import reset_run_cost, set_call_cost_hook
    reset_run_cost()
    _csym = "¥" if pricing_currency() == "RMB" else ""

    def _on_call_cost(label: str, call_cost: float, run_total: float) -> None:
        amount = "free" if not call_cost else f"{_csym}{call_cost:.4f}"
        ui.cost_line(f"{label} · {amount}", f"run {_csym}{run_total:.4f}")

    set_call_cost_hook(_on_call_cost)

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
            _tidy_empty_dirs(cfg)
            db.close()
        return

    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("_stop", True))
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("_stop", True))
    if cfg.test_mode:
        _mode = "dry-run" if cfg.x_dry_run else "LIVE posts to X"
        ui.info(f"🧪 test soak started — {_mode}, cycling fast"
                f"{f' (max {test_cycles} cycles)' if test_cycles else ''}. Ctrl-C to stop.")
    else:
        ui.info(f"bot started — posting every {cfg.post_interval_hours:.2f}h "
                f"(dry_run={cfg.x_dry_run}). Ctrl-C to stop.")

    _today = db.get_today_stats()
    if _today["cost_rmb"] or _today["ai_calls"]:
        _sym = "¥" if pricing_currency() == "RMB" else ""
        ui.info(f"today so far: {_today['ai_calls']} AI calls · {_sym}{_today['cost_rmb']:.4f}")

    throttles = errors = 0
    reauth_notified = reauth_pending   # already told them at startup; don't repeat it
    cycles = posts = 0
    reason = "interrupted"
    while not _stop:
        try:
            db.reset_stuck_posting()
            flush_pending(db, cfg, clients)
            posted_today = db.count_posts_rolling_24h()
            if cfg.test_mode or (_due_to_post(db, cfg) and posted_today < cfg.max_posts_per_day):
                outcome = run_one_cycle(db, cfg, ic, clients)
                logger.info("cycle: %s", outcome)
                if cfg.test_mode:
                    if outcome in ("exhausted", "ai_cap"):
                        reason = ("library exhausted" if outcome == "exhausted"
                                  else "daily AI cap reached")
                        break
                    if outcome == "posted":
                        posts += 1
                    cycles += 1
                    if test_cycles and cycles >= test_cycles:
                        reason = f"{cycles} cycles done"
                        break
                    ui.info(f"⏱  [TEST {cycles}{('/' + str(test_cycles)) if test_cycles else ''}] "
                            f"~{cfg.post_interval_hours:.2f}h wait skipped → next cycle now")
                elif outcome == "posted":
                    ui.ok(f"posted — holding for the next ~{cfg.post_interval_hours:.2f}h slot")
                elif outcome == "exhausted":
                    ui.info("no new photo to post right now")
                else:
                    ui.info(f"cycle complete — {outcome}")
            # When NOT due (or capped), the heartbeat below narrates the wait instead.
            throttles = errors = 0
            reauth_notified = False
        except ReauthRequired as exc:
            # A closed lid or dropped network can invalidate the iCloud session. Rebuild
            # it from the saved cookies before giving up — this recovers the common
            # "stale after sleep" case without ever stopping the bot.
            logger.warning("iCloud session lost (%s) — re-establishing …", exc)
            try:
                ic.ensure_session()
                logger.info("iCloud session re-established; resuming.")
                throttles = errors = 0
                reauth_notified = False
            except ReauthRequired as exc2:
                # genuinely needs interactive 2FA → notify once, then stay alive and keep
                # retrying so a later `ic2x login` (fresh cookies on disk) is picked up.
                if not reauth_notified:
                    _notify_reauth(cfg, exc2)
                    reauth_notified = True
                _sleep_interruptible(min(cfg.daemon_check_interval, 600))
            continue
        except PyiCloudThrottled as exc:
            throttles += 1
            backoff = min(cfg.daemon_check_interval * (2 ** throttles), 3600)
            logger.warning("throttled (%d) — backing off %ds: %s", throttles, backoff, exc)
            _sleep_interruptible(backoff)
            continue
        except Exception as exc:  # noqa: BLE001 — one bad cycle must not kill the bot
            # Transient failures (e.g. sockets dropped while the lid was closed) recover
            # fast: short, growing backoff instead of the full poll interval. After a few
            # in a row, also rebuild the iCloud session in case it went stale.
            errors += 1
            backoff = min(30 * (2 ** (errors - 1)), 300)
            logger.error("cycle error (%d) — retrying in %ds: %s", errors, backoff, exc,
                         exc_info=True)
            if errors >= 3:
                try:
                    ic.ensure_session()
                    logger.info("iCloud session refreshed after repeated errors.")
                except ReauthRequired as e2:
                    # Interactive 2FA is the one failure a fresh process CANNOT fix, so
                    # escalating to the re-exec below only burns the loop's reauth
                    # handling. Notify once and keep polling instead — `ic2x login` then
                    # revives this same process.
                    if not reauth_notified:
                        _notify_reauth(cfg, e2)
                        reauth_notified = True
                    _sleep_interruptible(min(cfg.daemon_check_interval, 600))
                    continue
                except Exception as e2:  # noqa: BLE001
                    logger.warning("session refresh failed: %s", e2)
            if errors >= 6:
                # In-process session rebuilds can fail while a FRESH process works
                # (observed 2026-07-15: pyicloud's mid-call session renewal kept
                # dying for hours; a new process recovered instantly). exec a clean
                # interpreter in place — same PID, same nohup/log redirection, the
                # DB carries all state, startup reconcile resyncs with X.
                logger.error("still failing after %d errors — re-execing a fresh "
                             "process to shed poisoned session state", errors)
                ui.warn("♻️  restarting the bot process (fresh iCloud session) …")
                import os as _os
                import sys as _sys
                _os.execv(_sys.executable, [_sys.executable, *_sys.argv])
            _sleep_interruptible(backoff)
            continue

        _sleep_interruptible(
            0 if cfg.test_mode else cfg.daemon_check_interval,
            heartbeat=None if cfg.test_mode else (lambda: _waiting_status(db, cfg)),
        )

    _tidy_empty_dirs(cfg)
    db.close()
    if cfg.test_mode:
        _label = "dry-posts" if cfg.x_dry_run else "LIVE posts"
        ui.ok(f"🧪 test run: {cycles} cycles · {posts} {_label} · stop: {reason}. "
              f"Review → {cfg.reviewed_dir}/ · logs → {cfg.logs_dir}/")
    else:
        ui.info("bot stopped.")
