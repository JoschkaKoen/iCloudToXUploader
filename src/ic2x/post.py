"""
Post approved images to X (Twitter).

Uses tweepy v1.1 API for media upload (v2 doesn't support it) and
tweepy v2 Client for tweet creation.

Ported from XBot-3/nodes/publish.py, adapted for photos instead of video.

Idempotency pattern:
  db.set_status(sha256, "posting")     # written BEFORE the API call
  create_tweet(...)
  db.set_status(sha256, "posted", ...) # written AFTER the API call

Any row stuck in "posting" on startup means the process was killed between
these two writes — run.py checks for this and exits with a warning.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import tweepy

from ic2x.config import Config, load_config, ensure_dirs
from ic2x.db import DB
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


def post() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    db = DB(cfg.db_path)
    approved = db.get_approved()

    if not approved:
        ui.info("No approved images to post.")
        db.close()
        return

    ui.post_section_banner(len(approved), cfg.x_dry_run)

    if not cfg.x_dry_run:
        api_v1, client_v2 = _build_clients(cfg)
    else:
        api_v1 = client_v2 = None

    posted_count = 0

    for row in approved:
        sha256   = row["sha256"]
        phash    = row["phash"] or ""
        caption  = row["caption"] or ""
        filename = phash + ".jpg"
        img_path = cfg.approved_dir / filename

        if not img_path.exists():
            ui.warn(f"Approved file not found on disk: {filename} — skipping")
            continue

        ui.post_banner(filename, caption, cfg.x_dry_run)

        if cfg.x_dry_run:
            ui.post_dry(filename, caption)
            posted_count += 1
            continue

        _post_succeeded = False
        try:
            tweet_id, tweet_url = _post_image(
                img_path, caption, sha256, db, cfg, api_v1, client_v2
            )
            _post_succeeded = True
            ui.post_ok(tweet_url)
            shutil.move(str(img_path), cfg.posted_dir / filename)
            # Move sidecar JSON if present
            json_src = cfg.approved_dir / (phash + ".json")
            if json_src.exists():
                shutil.move(str(json_src), cfg.posted_dir / json_src.name)
            db.increment_images_posted()
            posted_count += 1
        except Exception as exc:
            logger.error("Failed to post %s: %s", filename, exc, exc_info=True)
            ui.err(f"Failed to post {filename}: {exc}")
        finally:
            # Reset to 'approved' on any failure (including Ctrl+C / KeyboardInterrupt)
            # so the startup guard never blocks the next run.
            if not _post_succeeded:
                db.set_status(sha256, "approved")
                ui.info("Status reset to 'approved' — will retry on next `ic2x post`.")

    ui.post_summary(posted_count, cfg.x_dry_run)
    db.close()


@with_retry(max_attempts=4, base_delay=10.0, backoff=2.0, label="upload_image")
def _upload_image(api_v1, path: Path) -> int:
    logger.info("post: uploading %s", path.name)
    media = api_v1.media_upload(filename=str(path))
    logger.info("post: media uploaded, id=%s", media.media_id)
    return media.media_id


@with_retry(max_attempts=6, base_delay=15.0, backoff=2.0, label="create_tweet")
def _create_tweet_api(client_v2, text: str, media_id: int) -> tuple[str, str]:
    response = client_v2.create_tweet(text=text, media_ids=[media_id])
    tweet_id  = str(response.data["id"])
    tweet_url = f"https://x.com/i/web/status/{tweet_id}"
    logger.info("post: tweet posted → %s", tweet_url)
    return tweet_id, tweet_url


def _post_image(
    path: Path,
    caption: str,
    sha256: str,
    db: DB,
    cfg: Config,
    api_v1,
    client_v2,
) -> tuple[str, str]:
    media_id = _upload_image(api_v1, path)

    db.set_status(sha256, "posting")   # idempotency anchor — written BEFORE tweet

    tweet_id, tweet_url = _create_tweet_api(client_v2, caption, media_id)

    db.set_status(
        sha256, "posted",
        tweet_id=tweet_id,
        posted_at=datetime.now(timezone.utc).isoformat(),
    )
    db.set_last_posted_at(datetime.now(timezone.utc))
    return tweet_id, tweet_url
