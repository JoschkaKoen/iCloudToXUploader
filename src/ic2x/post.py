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

Multi-image posts:
  When GROUP_HAMMING_THRESHOLD > 0, approved images whose pHash Hamming
  distance is ≤ the threshold are bundled into a single tweet (up to 4 images).
  All members are set to "posting" before the tweet and "posted" after.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import tweepy

from ic2x.config import Config, load_config, ensure_dirs
from ic2x.group import cluster_by_phash
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


def _group_approved(approved: list, threshold: int) -> list[list]:
    return cluster_by_phash(approved, threshold)


def post() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    db = DB(cfg.db_path)
    approved = db.get_approved()

    if not approved:
        ui.info("No approved images to post.")
        db.close()
        return

    groups = _group_approved(approved, cfg.group_hamming_threshold)
    ui.post_section_banner(len(approved), cfg.x_dry_run)

    if not cfg.x_dry_run:
        api_v1, client_v2 = _build_clients(cfg)
    else:
        api_v1 = client_v2 = None

    posted_count = 0

    for group in groups:
        if len(group) == 1:
            # ── Single-image path (unchanged) ────────────────────────────────
            row      = group[0]
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
                json_src = cfg.approved_dir / (phash + ".json")
                if json_src.exists():
                    shutil.move(str(json_src), cfg.posted_dir / json_src.name)
                db.increment_images_posted()
                posted_count += 1
            except Exception as exc:
                logger.error("Failed to post %s: %s", filename, exc, exc_info=True)
                ui.err(f"Failed to post {filename}: {exc}")
            finally:
                if not _post_succeeded:
                    db.set_status(sha256, "approved")
                    ui.info("Status reset to 'approved' — will retry on next `ic2x post`.")

        else:
            # ── Multi-image path ─────────────────────────────────────────────
            shas      = [r["sha256"] for r in group]
            caption   = group[0]["caption"] or ""
            filenames = [(r["phash"] or "") + ".jpg" for r in group]
            img_paths = [cfg.approved_dir / fn for fn in filenames]

            missing = [p.name for p in img_paths if not p.exists()]
            if missing:
                ui.warn(f"Group missing file(s) on disk: {missing} — skipping group")
                continue

            ui.post_banner(f"{len(group)} images", caption, cfg.x_dry_run)

            if cfg.x_dry_run:
                for fn in filenames:
                    ui.post_dry(fn, caption)
                posted_count += len(group)
                continue

            _post_succeeded = False
            try:
                tweet_id, tweet_url = _post_group(
                    img_paths, caption, shas, db, api_v1, client_v2
                )
                _post_succeeded = True
                ui.post_ok(tweet_url)
                for member, img_path in zip(group, img_paths):
                    phash = member["phash"] or ""
                    shutil.move(str(img_path), cfg.posted_dir / img_path.name)
                    json_src = cfg.approved_dir / (phash + ".json")
                    if json_src.exists():
                        shutil.move(str(json_src), cfg.posted_dir / json_src.name)
                    db.increment_images_posted()
                posted_count += len(group)
            except Exception as exc:
                logger.error("Failed to post group %s: %s", filenames, exc, exc_info=True)
                ui.err(f"Failed to post group: {exc}")
            finally:
                if not _post_succeeded:
                    for sha in shas:
                        db.set_status(sha, "approved")
                    ui.info("Group status reset to 'approved' — will retry on next `ic2x post`.")

    ui.post_summary(posted_count, cfg.x_dry_run)
    db.close()


@with_retry(max_attempts=4, base_delay=10.0, backoff=2.0, label="upload_image")
def _upload_image(api_v1, path: Path) -> int:
    logger.info("post: uploading %s", path.name)
    media = api_v1.media_upload(filename=str(path))
    logger.info("post: media uploaded, id=%s", media.media_id)
    return media.media_id


@with_retry(max_attempts=6, base_delay=15.0, backoff=2.0, label="create_tweet")
def _create_tweet_api(client_v2, text: str, media_ids: list[int]) -> tuple[str, str]:
    response = client_v2.create_tweet(text=text, media_ids=media_ids)
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

    tweet_id, tweet_url = _create_tweet_api(client_v2, caption, [media_id])

    db.set_status(
        sha256, "posted",
        tweet_id=tweet_id,
        posted_at=datetime.now(timezone.utc).isoformat(),
    )
    db.set_last_posted_at(datetime.now(timezone.utc))
    return tweet_id, tweet_url


def _post_group(
    paths: list[Path],
    caption: str,
    sha256s: list[str],
    db: DB,
    api_v1,
    client_v2,
) -> tuple[str, str]:
    media_ids = [_upload_image(api_v1, p) for p in paths]

    # Idempotency anchor: mark ALL group members before the tweet
    for sha in sha256s:
        db.set_status(sha, "posting")

    tweet_id, tweet_url = _create_tweet_api(client_v2, caption, media_ids)

    posted_at = datetime.now(timezone.utc).isoformat()
    for sha in sha256s:
        db.set_status(sha, "posted", tweet_id=tweet_id, posted_at=posted_at)
    db.set_last_posted_at(datetime.now(timezone.utc))
    return tweet_id, tweet_url
