"""
Startup reconciliation — bring the local DB in line with what's actually live on X.

Runs once at the start of the normal bot (`bot()`), live runs only: take the DB's
most-recent posted tweet ids and check each one's existence on X with Bearer-token
`get_tweets(ids=…)` — the only X read that works here (OAuth 1.0a 401s on tweet reads,
and the user-timeline endpoint misbehaves). Any tracked post X reports as
"resource-not-found" was deleted by the owner → RETIRE that photo: mark the row rejected
and keep the asset in the seen-set, so it is never posted again. Deleting a post is the
owner's verdict on the photo; until 2026-08-04 this re-queued it instead, so deleting a
post you disliked simply put it back in the pool. Captions of still-live posts are
filled if missing.

FAIL-OPEN: any X/network/tier error logs a note and returns — startup is never blocked.
A sanity cap skips retiring when implausibly many look deleted (a bad/partial read).
"""

from __future__ import annotations

import logging
import time

from ic2x.config import Config
from ic2x.db import DB
from ic2x.utils import ui
from ic2x.utils.decision_log import log_decision

logger = logging.getLogger("ic2x.reconcile")

_X_READ_ATTEMPTS = 3        # one initial try + 2 quick retries before giving up (fail-open)
_X_READ_RETRY_DELAY = 2.0   # seconds between attempts — close succession, for a transient blip


def reconcile_with_x(db: DB, cfg: Config, lookup_tweets) -> None:
    """`lookup_tweets(ids) -> (live: dict[id->text], deleted: set[id])` reports which
    of the given tweet ids still exist on X. Fail-open."""
    candidates = db.posted_for_reconcile(cfg.reconcile_recent_n)
    if not candidates:
        return  # nothing posted yet → nothing to reconcile
    ui.info(f"reconcile — checking {len(candidates)} recent posts against X …")
    ids = [str(c["tweet_id"]) for c in candidates]

    # Retry the X read a couple of times in close succession before giving up — a brief
    # proxy/VPN blip shouldn't skip reconcile. Fail-open only after all attempts fail.
    for attempt in range(1, _X_READ_ATTEMPTS + 1):
        try:
            live, deleted = lookup_tweets(ids)
            break
        except Exception as exc:  # noqa: BLE001 — never block startup on an X read
            if attempt < _X_READ_ATTEMPTS:
                ui.warn(f"reconcile — X read failed ({attempt}/{_X_READ_ATTEMPTS}), retrying …")
                time.sleep(_X_READ_RETRY_DELAY)
            else:
                ui.warn(f"reconcile — skipped after {_X_READ_ATTEMPTS} tries (X read failed: {exc})")
                return

    # `deleted` = ids X reports as resource-not-found. Anything not in `deleted` still
    # EXISTS — whether viewable (`live`) or restricted (e.g. visibility lowered) — so
    # it counts as still-on-X and is NOT retired.
    deletions = [c for c in candidates if str(c["tweet_id"]) in deleted]
    ui.info(f"reconcile — {len(candidates) - len(deletions)} still on X · "
            f"{len(deletions)} deleted")

    # Sanity guard: a bad/partial read makes most look deleted — don't act.
    if len(deletions) > cfg.reconcile_max_requeue_per_run:
        ui.warn(f"reconcile — {len(deletions)} look deleted (cap "
                f"{cfg.reconcile_max_requeue_per_run}); likely an X read glitch — "
                "skipping retire this run")
        return

    # Fill missing captions for still-live posts (never overwrite an existing one).
    filled = 0
    for c in candidates:
        if str(c["tweet_id"]) in live and not (c["caption"] or "").strip():
            try:
                db.update_caption(c["id"], live[str(c["tweet_id"])])
                filled += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("reconcile: caption fill failed for %s: %s", c["id"], exc)

    # Retire each deleted post (one bad row never aborts the rest). The owner deleting
    # a post is a verdict on that photo, so it is recorded as rejected and stays in the
    # seen-set — never posted again. It used to be re-queued, which meant deleting a
    # post you disliked just put it back in the pool to return later.
    retired = 0
    for c in deletions:
        try:
            log_decision(cfg.logs_dir, outcome="deleted_on_x", asset_id=c["asset_id"],
                         detail={"tweet_id": c["tweet_id"], "caption": c["caption"]})
            db.reject_deleted(c["id"], c["asset_id"])
            retired += 1
            snippet = (c["caption"] or "").replace("\n", " ").strip()[:60] or "(no caption)"
            ui.info(f'   ↳ deleted on X → retired, will not post again: "{snippet}"')
        except Exception as exc:  # noqa: BLE001
            logger.warning("reconcile: retire failed for image %s: %s", c["id"], exc)

    # Keep the 5h post timer in sync with the most-recent STILL-LIVE post EVERY run — a
    # post deleted on X (whether this run or a previous one) must never keep holding the
    # interval. Normally a no-op (timer already == newest live post); corrects staleness.
    db.refresh_last_posted_at()

    bits = [f"{retired} retired" if retired else "all in sync"]
    if filled:
        bits.append(f"{filled} caption{'s' if filled != 1 else ''} filled")
    ui.ok("reconcile done — " + " · ".join(bits))


def make_x_lookup(cfg: Config):
    """The real id-existence checker: Bearer-token `get_tweets(ids=…)` in batches of
    100. Returns (live{id->text}, deleted{id}); deleted = ids X reports as not-found."""
    import tweepy

    reader = tweepy.Client(bearer_token=cfg.twitter_bearer_token)

    def lookup(ids):
        live: dict[str, str] = {}
        deleted: set[str] = set()
        for i in range(0, len(ids), 100):
            resp = reader.get_tweets(ids=ids[i:i + 100], tweet_fields=["text"])
            for t in (resp.data or []):
                live[str(t.id)] = t.text or ""
            for e in (resp.errors or []):
                if e.get("resource_type") == "tweet" and "resource-not-found" in str(e.get("type", "")):
                    deleted.add(str(e.get("resource_id")))
        return live, deleted

    return lookup
