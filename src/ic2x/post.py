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


def _post_image(path: Path, caption: str, sha256: str, db: DB, api_v1, client_v2) -> tuple[str, str]:
    media_id = _upload_image(api_v1, path)
    db.set_status(sha256, Status.POSTING)   # idempotency anchor — BEFORE the tweet
    tweet_id, tweet_url = _create_tweet_api(client_v2, caption, [media_id])
    db.set_status(
        sha256, Status.POSTED,
        tweet_id=tweet_id, posted_at=datetime.now(timezone.utc).isoformat(),
    )
    db.set_last_posted_at(datetime.now(timezone.utc))
    return tweet_id, tweet_url


def _save_scene_thumb(src: Path, phash: str, cfg: Config) -> None:
    """Write a small scene_thumbs/{phash}.jpg for the same-scene dedup set.
    Best-effort — never raises, never blocks a post."""
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
    except Exception as exc:  # noqa: BLE001 — dedup aid, never fatal
        logger.warning("scene_thumb: could not save %s: %s", phash, exc)


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
            _, tweet_url = _post_image(img_path, caption, sha256, db, api_v1, client_v2)
            ui.post_ok(tweet_url)

        _save_scene_thumb(img_path, phash, cfg)  # small thumb for same-scene dedup
        dest = cfg.posted_dir / filename
        try:
            shutil.move(str(img_path), str(dest))
        except Exception as exc:  # noqa: BLE001 — file housekeeping, not fatal
            logger.warning("post_one: could not move %s to posted/: %s", filename, exc)
        db.increment_images_posted()
        return True

    except Exception as exc:  # noqa: BLE001
        logger.error("post_one: failed to post %s: %s", filename, exc, exc_info=True)
        attempts = db.incr_post_attempts(sha256)
        if attempts >= cfg.post_max_attempts:
            db.set_status(sha256, Status.REJECTED, reject_stage="post_failed",
                          reject_reason=str(exc)[:200])
            ui.err(f"post failed permanently after {attempts} attempts: {filename}")
        else:
            db.set_status(sha256, Status.APPROVED)
            ui.warn(f"post failed (attempt {attempts}/{cfg.post_max_attempts}): {filename}")
        return False
