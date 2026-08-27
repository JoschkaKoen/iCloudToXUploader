"""
Post an approved image to X (Twitter).

tweepy v1.1 uploads the media (v2 can't); tweepy v2 creates the tweet.

Idempotency: db.set_status(sha, "posting") is written BEFORE create_tweet and
"posted" after. A row stuck in "posting" means the process died between the two
writes — the bot's reset_stuck_posting() recovers it to "approved" at cycle start.

In X_DRY_RUN the state machine still advances (image → posted, tweet_id="DRYRUN",
last_posted_at set) so a burn-in exercises the full loop without a real tweet.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests
import tweepy

from ic2x.config import Config
from ic2x.db import DB
from ic2x.status import Status
from ic2x.utils import ui
from ic2x.utils.retry import with_retry

logger = logging.getLogger("ic2x.post")


def _build_clients(cfg: Config):
    auth = tweepy.OAuth1UserHandler(
        consumer_key=cfg.twitter_consumer_key,
        consumer_secret=cfg.twitter_consumer_secret,
        access_token=cfg.twitter_access_token,
        access_token_secret=cfg.twitter_access_token_secret,
    )
    api_v1 = tweepy.API(auth)
    client_v2 = tweepy.Client(
        consumer_key=cfg.twitter_consumer_key,
        consumer_secret=cfg.twitter_consumer_secret,
        access_token=cfg.twitter_access_token,
        access_token_secret=cfg.twitter_access_token_secret,
    )
    return api_v1, client_v2


def make_clients(cfg: Config):
    """Return (api_v1, client_v2), or (None, None) in dry-run."""
    if cfg.x_dry_run:
        return None, None
    return _build_clients(cfg)


@with_retry(max_attempts=4, base_delay=10.0, backoff=2.0, label="upload_image")
def _upload_image(api_v1, path: Path) -> int:
    logger.info("post: uploading %s", path.name)
    media = api_v1.media_upload(filename=str(path))
    logger.info("post: media uploaded, id=%s", media.media_id)
    return media.media_id


@with_retry(max_attempts=6, base_delay=15.0, backoff=2.0, label="create_tweet")
def _create_tweet_api(client_v2, text: str, media_ids: list[int]) -> tuple[str, str]:
    response = client_v2.create_tweet(text=text, media_ids=media_ids)
    tweet_id = str(response.data["id"])
    tweet_url = f"https://x.com/i/web/status/{tweet_id}"
    logger.info("post: tweet posted → %s", tweet_url)
    return tweet_id, tweet_url


def _set_alt_text(api_v1, media_id: int, alt: str) -> None:
    """Attach alt text to the uploaded image. Free discoverability (X indexes it) and
    basic accessibility. The judge already produced a one-line description of the photo
    (`shows`), so this costs no extra AI call. Best-effort: a post must never fail
    because a metadata call did."""
    alt = " ".join((alt or "").split())[:1000]     # X caps alt text at 1000 chars
    if not alt:
        return
    try:
        api_v1.create_media_metadata(media_id, alt)
        logger.info("post: alt text set (%d chars)", len(alt))
    except Exception as exc:  # noqa: BLE001
        logger.warning("post: alt text failed (posting anyway): %s", str(exc)[:160])


def _post_image(path: Path, caption: str, sha256: str, db: DB, api_v1, client_v2,
                alt_text: str = "") -> tuple[str, str]:
    media_id = _upload_image(api_v1, path)
    _set_alt_text(api_v1, media_id, alt_text)
    db.set_status(sha256, Status.POSTING)   # idempotency anchor — BEFORE the tweet
    tweet_id, tweet_url = _create_tweet_api(client_v2, caption, [media_id])
    db.set_status(
        sha256, Status.POSTED,
        tweet_id=tweet_id, posted_at=datetime.now(timezone.utc).isoformat(),
    )
    db.set_last_posted_at(datetime.now(timezone.utc))
    return tweet_id, tweet_url


def _prune_scene_thumbs(scene_dir: Path, keep: int) -> None:
    """Keep only the `keep` newest scene thumbs (by mtime). Same-scene dedup only ever
    looks up the few most-recently-posted phashes, so without this the dir would grow
    one file per post forever on a long-running bot. Best-effort."""
    try:
        thumbs = sorted(scene_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in thumbs[keep:]:
            p.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 — housekeeping, never fatal
        logger.warning("scene_thumb: prune failed: %s", exc)


def _save_scene_thumb(src: Path, phash: str, cfg: Config) -> None:
    """Write a small scene_thumbs/{phash}.jpg for the same-scene dedup set, then prune
    old thumbs so the dir stays bounded. Best-effort — never raises, never blocks a post."""
    if not phash:
        return
    try:
        from PIL import Image

        from ic2x.utils.image_utils import oriented
        max_px = cfg.scene_dedup_image_max_px or 384
        cfg.scene_thumbs_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = oriented(im)
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.thumbnail((max_px, max_px))
            im.save(cfg.scene_thumbs_dir / f"{phash}.jpg", "JPEG", quality=85)
        # keep a generous multiple of the dedup window (which is only ~6) so the lookup
        # set is always present, while bounding unlimited growth.
        _prune_scene_thumbs(cfg.scene_thumbs_dir, keep=max(50, getattr(cfg, "scene_dedup_recent_n", 6) * 6))
    except Exception as exc:  # noqa: BLE001 — dedup aid, never fatal
        logger.warning("scene_thumb: could not save %s: %s", phash, exc)


# Connectivity/availability hiccups (X unreachable, DNS/SSL/timeout, 5xx, rate limit)
# vs. a genuine rejection of THIS photo. The former must NEVER burn the post-attempt
# budget — the laptop sleeps, the VPN drops, DNS blips; the photo should still post once
# X is reachable again. (These surface only after _upload_image already retried 4×.)
_TRANSIENT_TWEEPY = (tweepy.TooManyRequests, tweepy.TwitterServerError)
_NETWORK_SIGNATURES = (
    "failed to send request", "max retries exceeded", "connection aborted",
    "connection reset", "connectionerror", "connecttimeout", "read timed out",
    "readtimeout", "nameresolution", "name resolution", "sslerror", "timed out",
    "network is unreachable", "temporary failure in name resolution",
)


def _is_transient_post_error(exc: BaseException) -> bool:
    """True when posting failed because X was unreachable (network/DNS/SSL/timeout, 5xx,
    or 429) rather than because this photo was rejected. Transient errors DEFER the post
    to a later cycle instead of permanently dropping a good photo."""
    if isinstance(exc, _TRANSIENT_TWEEPY):
        return True
    # tweepy re-raises requests' connection errors as TweepyException("Failed to send …"),
    # chaining the original on __cause__/__context__.
    for e in (exc, getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        if isinstance(e, requests.exceptions.RequestException):
            return True
    msg = str(exc).lower()
    return any(sig in msg for sig in _NETWORK_SIGNATURES)


def post_one(row, cfg: Config, db: DB, api_v1, client_v2) -> bool:
    """Post one APPROVED row (file at approved_dir/{phash}.jpg). Returns True on
    success. On failure: bumps post_attempts; once >= cfg.post_max_attempts marks
    REJECTED(post_failed), else resets to APPROVED for the next flush."""
    sha256 = row["sha256"]
    phash = row["phash"] or ""
    caption = row["caption"] or ""
    filename = f"{phash}.jpg"
    img_path = cfg.approved_dir / filename

    if not img_path.exists():
        logger.warning("post_one: approved file missing on disk: %s", filename)
        db.set_status(sha256, Status.REJECTED, reject_stage="post_failed",
                      reject_reason="approved file missing on disk")
        return False

    try:
        if cfg.x_dry_run:
            ui.post_dry(filename, caption)
            db.set_status(
                sha256, Status.POSTED, tweet_id="DRYRUN", caption=caption,
                posted_at=datetime.now(timezone.utc).isoformat(),
            )
            db.set_last_posted_at(datetime.now(timezone.utc))
        else:
            alt = ""
            try:
                alt = row["alt_text"] or ""
            except (KeyError, IndexError, TypeError):
                alt = ""      # older rows / lightweight test fakes carry no alt column
            _, tweet_url = _post_image(img_path, caption, sha256, db, api_v1, client_v2,
                                       alt_text=alt)
            ui.post_ok(tweet_url)

        _save_scene_thumb(img_path, phash, cfg)  # small thumb for same-scene dedup
        cfg.posted_dir.mkdir(parents=True, exist_ok=True)
        dest = cfg.posted_dir / filename
        try:
            shutil.move(str(img_path), str(dest))
        except Exception as exc:  # noqa: BLE001 — file housekeeping, not fatal
            logger.warning("post_one: could not move %s to posted/: %s", filename, exc)
        db.increment_images_posted()
        return True

    except Exception as exc:  # noqa: BLE001
        if _is_transient_post_error(exc):
            # X unreachable (laptop asleep, VPN/DNS/SSL hiccup) — NOT this photo's fault.
            # Keep it APPROVED so flush_pending retries it next cycle; never let a network
            # blip burn the attempt budget and permanently drop a good photo.
            db.set_status(sha256, Status.APPROVED)
            logger.warning("post_one: X unreachable for %s — deferring: %s",
                           filename, str(exc)[:160])
            ui.warn(f"post deferred — X unreachable, will retry next cycle: {filename}")
            return False
        logger.error("post_one: failed to post %s: %s", filename, exc, exc_info=True)
        attempts = db.incr_post_attempts(sha256)
        if attempts >= cfg.post_max_attempts:
            db.set_status(sha256, Status.REJECTED, reject_stage="post_failed",
                          reject_reason=str(exc)[:200])
            try:
                img_path.unlink(missing_ok=True)  # don't leave the prepared file as approved/ clutter
            except OSError as err:
                logger.warning("post_one: could not remove rejected %s: %s", filename, err)
            ui.err(f"post failed permanently after {attempts} attempts: {filename}")
        else:
            db.set_status(sha256, Status.APPROVED)
            ui.warn(f"post failed (attempt {attempts}/{cfg.post_max_attempts}): {filename}")
        return False
