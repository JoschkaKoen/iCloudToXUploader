"""
Startup reconciliation — bring the local DB in line with what's actually live on X.

Runs once at the start of the normal bot (`bot()`), live runs only: take the DB's
most-recent posted tweet ids and check each one's existence on X with Bearer-token
`get_tweets(ids=…)` — the only X read that works here (OAuth 1.0a 401s on tweet reads,
and the user-timeline endpoint misbehaves). Any tracked post X reports as
"resource-not-found" was deleted by the owner → re-queue the iCloud photo (delete the
stale row + clear the asset from the seen-set) so the bot re-posts it. Captions of
still-live posts are filled if missing.

FAIL-OPEN: any X/network/tier error logs a note and returns — startup is never blocked.
A sanity cap skips re-queuing when implausibly many look deleted (a bad/partial read).
"""

from __future__ import annotations

import logging

from ic2x.config import Config
from ic2x.db import DB
from ic2x.utils import ui
from ic2x.utils.decision_log import log_decision

logger = logging.getLogger("ic2x.reconcile")


def reconcile_with_x(db: DB, cfg: Config, lookup_tweets) -> None:
    """`lookup_tweets(ids) -> (live: dict[id->text], deleted: set[id])` reports which
    of the given tweet ids still exist on X. Fail-open."""
    candidates = db.posted_for_reconcile(cfg.reconcile_recent_n)
    if not candidates:
        return  # nothing posted yet → nothing to reconcile
    ui.info(f"reconcile — checking {len(candidates)} recent posts against X …")
    ids = [str(c["tweet_id"]) for c in candidates]
    try:
        live, deleted = lookup_tweets(ids)
    except Exception as exc:  # noqa: BLE001 — never block startup on an X read
        ui.warn(f"reconcile — skipped (X read failed: {exc})")
        return

    # `deleted` = ids X reports as resource-not-found. Anything not in `deleted` still
    # EXISTS — whether viewable (`live`) or restricted (e.g. visibility lowered) — so
    # it counts as still-on-X and is NOT re-queued.
    deletions = [c for c in candidates if str(c["tweet_id"]) in deleted]
    ui.info(f"reconcile — {len(candidates) - len(deletions)} still on X · "
            f"{len(deletions)} deleted")

    # Sanity guard: a bad/partial read makes most look deleted — don't act.
    if len(deletions) > cfg.reconcile_max_requeue_per_run:
        ui.warn(f"reconcile — {len(deletions)} look deleted (cap "
                f"{cfg.reconcile_max_requeue_per_run}); likely an X read glitch — "
                "skipping re-queue this run")
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

    # Re-queue each deleted post (one bad row never aborts the rest).
    requeued = 0
    for c in deletions:
        try:
            log_decision(cfg.logs_dir, outcome="deleted_on_x", asset_id=c["asset_id"],
                         detail={"tweet_id": c["tweet_id"], "caption": c["caption"]})
            db.requeue_deleted(c["id"], c["asset_id"])
            requeued += 1
            snippet = (c["caption"] or "").replace("\n", " ").strip()[:60] or "(no caption)"
            ui.info(f'   ↳ deleted on X → re-queued: "{snippet}"')
        except Exception as exc:  # noqa: BLE001
            logger.warning("reconcile: re-queue failed for image %s: %s", c["id"], exc)

    bits = [f"{requeued} re-queued" if requeued else "all in sync"]
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
