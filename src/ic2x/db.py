"""
All SQLite access lives here. Nothing else touches the DB directly.

Two tables drive the bot:
  • asset_index — one row per still-image iCloud asset (synced from metadata).
    `seen` is the single source of truth for "the bot has already decided on
    this asset" (winner, loser, screenshot, or error) — it is what stops a
    burst from ever being re-assembled. `attempts` is the poison-burst breaker.
  • images — detailed record for assets that reached the judge as a winner
    candidate (keyed by sha256, since only winners are downloaded full-res).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from ic2x.status import Status


_SCHEMA = """
-- Decided-asset tracker (the seen-set). Rows are created on-demand when a burst
-- is committed; ordering comes from live recently_added() iteration, not here.
CREATE TABLE IF NOT EXISTS asset_index (
    asset_id    TEXT PRIMARY KEY,
    seen        INTEGER DEFAULT 0,      -- 1 once the bot has decided on it
    attempts    INTEGER DEFAULT 0,      -- pre-commit failures (poison-burst breaker)
    decided_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id        TEXT,
    sha256          TEXT UNIQUE NOT NULL,
    phash           TEXT,
    source_filename TEXT,
    status          TEXT NOT NULL,
    reject_stage    TEXT,
    reject_reason   TEXT,
    safety_raw      TEXT,
    quality_raw     TEXT,
    caption         TEXT,
    tweet_id        TEXT,
    post_attempts   INTEGER DEFAULT 0,
    processed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    posted_at       TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_images_asset ON images(asset_id);

CREATE TABLE IF NOT EXISTS run_stats (
    date            TEXT PRIMARY KEY,
    ai_calls        INTEGER DEFAULT 0,  -- DECISION calls (judge/owner/rotation/caption)
    images_posted   INTEGER DEFAULT 0,
    cost_rmb        REAL DEFAULT 0,     -- estimated daily AI spend (tokens + any flat API)
    color_calls     INTEGER DEFAULT 0,  -- VIAPI color-enhance calls (for the monthly free quota)
    support_calls   INTEGER DEFAULT 0   -- cheap scene-grouping/dedup calls made while assembling
);

CREATE TABLE IF NOT EXISTS run_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Capture-date catalog of the WHOLE iCloud library (built once by a metadata
-- sweep, refreshed incrementally). Drives the chronological walk-back: iterate
-- created DESC, skipping seen assets; rank is the asset's position in the
-- ascending .all album at catalog time (positional re-fetch hint — live assets
-- can't be resolved by id, that hangs).
CREATE TABLE IF NOT EXISTS asset_catalog (
    asset_id     TEXT PRIMARY KEY,
    created      TEXT,               -- ISO capture datetime (UTC)
    rank         INTEGER,            -- position in ascending .all at catalog time
    cataloged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_catalog_created ON asset_catalog(created DESC);
"""


class DB:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        # Create the parent dir so opening a DB under a not-yet-created content dir
        # (e.g. `ic2x compare`'s throwaway work/compare_tmp.db before the bot has ever
        # created work/) doesn't fail with "unable to open database file".
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns to pre-existing tables (CREATE TABLE IF NOT EXISTS won't)."""
        for col, ddl in (("cost_rmb", "REAL DEFAULT 0"), ("color_calls", "INTEGER DEFAULT 0"),
                         ("support_calls", "INTEGER DEFAULT 0")):
            try:
                self._conn.execute(f"ALTER TABLE run_stats ADD COLUMN {col} {ddl}")
            except sqlite3.OperationalError:
                pass  # column already present

    def close(self) -> None:
        self._conn.close()

    # ── Seen-set (asset-level "decided") ───────────────────────────────────────

    def seen_asset_id(self, asset_id: str) -> bool:
        """True if the bot has already decided on this asset (any verdict)."""
        return self._conn.execute(
            "SELECT 1 FROM asset_index WHERE asset_id = ? AND seen = 1", (asset_id,)
        ).fetchone() is not None

    def incr_asset_attempts(self, asset_id: str) -> int:
        """Bump and return the pre-commit attempt count for a burst-head asset
        (upserts the row, since assets are only tracked on decision)."""
        self._conn.execute(
            "INSERT INTO asset_index (asset_id, attempts) VALUES (?, 1) "
            "ON CONFLICT(asset_id) DO UPDATE SET attempts = attempts + 1", (asset_id,)
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT attempts FROM asset_index WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        return row["attempts"] if row else 0

    # ── Burst commit (atomic) ──────────────────────────────────────────────────

    def commit_burst(
        self, seen_asset_ids: Sequence[str], winner: dict[str, Any] | None = None
    ) -> None:
        """Atomically mark every burst member decided (upsert seen=1), and
        optionally insert the winner's images row. A crash before this leaves zero
        state → the identical burst re-assembles; a crash after → flush_pending
        finishes the post.

        winner keys: asset_id, sha256, phash, filename, status (Status|str),
        caption, reject_stage, reject_reason.
        """
        try:
            self._conn.execute("BEGIN")
            for aid in seen_asset_ids:
                self._conn.execute(
                    "INSERT INTO asset_index (asset_id, seen) VALUES (?, 1) "
                    "ON CONFLICT(asset_id) DO UPDATE SET seen = 1", (aid,)
                )
            if winner is not None:
                status = winner["status"]
                status_value = status.value if isinstance(status, Status) else status
                reason = winner.get("reject_reason")
                if isinstance(reason, (dict, list)):
                    reason = json.dumps(reason)
                self._conn.execute(
                    "INSERT OR IGNORE INTO images "
                    "(asset_id, sha256, phash, source_filename, status, caption, "
                    " reject_stage, reject_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        winner.get("asset_id"), winner["sha256"], winner.get("phash"),
                        winner.get("filename"), status_value, winner.get("caption"),
                        winner.get("reject_stage"), reason,
                    ),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ── Image lookups / dedup ──────────────────────────────────────────────────

    def seen_sha256(self, sha256: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM images WHERE sha256 = ?", (sha256,)
        ).fetchone() is not None

    def get_image_by_sha(self, sha256: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM images WHERE sha256 = ?", (sha256,)
        ).fetchone()

    def seen_phash_similar(self, phash: str, threshold: int) -> bool:
        """True if any non-rejected kept image is within Hamming `threshold`.
        O(N) scan; fine for the modest number of kept/posted rows this bot makes."""
        import imagehash
        target = imagehash.hex_to_hash(phash)
        rows = self._conn.execute(
            "SELECT phash FROM images WHERE status != ? AND phash IS NOT NULL",
            (Status.REJECTED.value,),
        ).fetchall()
        for row in rows:
            try:
                if (target - imagehash.hex_to_hash(row["phash"])) <= threshold:
                    return True
            except Exception:
                continue
        return False

    def set_status(self, sha256: str, status: Status | str, **kwargs: Any) -> None:
        status_value = status.value if isinstance(status, Status) else status
        allowed = {
            "reject_stage", "reject_reason", "safety_raw", "quality_raw",
            "caption", "tweet_id", "posted_at", "phash", "asset_id", "post_attempts",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        for k, v in updates.items():
            if isinstance(v, (dict, list)):
                updates[k] = json.dumps(v)

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values())
        if set_clause:
            self._conn.execute(
                f"UPDATE images SET status = ?, {set_clause} WHERE sha256 = ?",
                [status_value] + values + [sha256],
            )
        else:
            self._conn.execute(
                "UPDATE images SET status = ? WHERE sha256 = ?", (status_value, sha256),
            )
        self._conn.commit()

    def incr_post_attempts(self, sha256: str) -> int:
        self._conn.execute(
            "UPDATE images SET post_attempts = post_attempts + 1 WHERE sha256 = ?", (sha256,)
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT post_attempts FROM images WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return row["post_attempts"] if row else 0

    # ── Posting recovery ───────────────────────────────────────────────────────

    def get_stuck_posting(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM images WHERE status = ?", (Status.POSTING.value,)
        ).fetchall()

    def reset_stuck_posting(self) -> int:
        cur = self._conn.execute(
            "UPDATE images SET status = ? WHERE status = ?",
            (Status.APPROVED.value, Status.POSTING.value),
        )
        self._conn.commit()
        return cur.rowcount

    def get_approved(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM images WHERE status = ? ORDER BY id ASC", (Status.APPROVED.value,)
        ).fetchall()

    # ── AI call budget + post cap ──────────────────────────────────────────────

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _ensure_today_stats(self) -> None:
        self._conn.execute("INSERT OR IGNORE INTO run_stats (date) VALUES (?)", (self._today(),))
        self._conn.commit()

    def check_daily_ai_limit(self, limit: int) -> bool:
        self._ensure_today_stats()
        row = self._conn.execute(
            "SELECT ai_calls FROM run_stats WHERE date = ?", (self._today(),)
        ).fetchone()
        return (row["ai_calls"] if row else 0) >= limit

    def check_daily_support_limit(self, limit: int) -> bool:
        """Runaway guard for the cheap assembly calls. Separate from the decision
        budget on purpose: scene-grouping fires at every pHash boundary, so billing it
        to the same meter as judging starves the walk-back — measured 2026-07-17, 172
        of 195 calls were grouping and only 19 bursts ever got judged."""
        self._ensure_today_stats()
        row = self._conn.execute(
            "SELECT COALESCE(support_calls, 0) AS n FROM run_stats WHERE date = ?",
            (self._today(),)
        ).fetchone()
        return (row["n"] if row else 0) >= limit

    def increment_ai_calls(self, n: int = 1) -> None:
        self._ensure_today_stats()
        self._conn.execute(
            "UPDATE run_stats SET ai_calls = ai_calls + ? WHERE date = ?", (n, self._today())
        )
        self._conn.commit()

    def increment_support_calls(self, n: int = 1) -> None:
        self._ensure_today_stats()
        self._conn.execute(
            "UPDATE run_stats SET support_calls = COALESCE(support_calls, 0) + ? "
            "WHERE date = ?", (n, self._today())
        )
        self._conn.commit()

    def increment_images_posted(self, n: int = 1) -> None:
        self._ensure_today_stats()
        self._conn.execute(
            "UPDATE run_stats SET images_posted = images_posted + ? WHERE date = ?",
            (n, self._today()),
        )
        self._conn.commit()

    def add_run_cost(self, rmb: float) -> None:
        """Accumulate today's estimated AI spend (RMB). Source-agnostic sink — token
        costs (compute_cost) and any flat per-call costs (e.g. VIAPI enhancement) both
        feed here, so `ic2x cost` shows the bot's true running spend."""
        if not rmb:
            return
        self._ensure_today_stats()
        self._conn.execute(
            "UPDATE run_stats SET cost_rmb = COALESCE(cost_rmb, 0) + ? WHERE date = ?",
            (rmb, self._today()),
        )
        self._conn.commit()

    def get_today_stats(self) -> dict[str, float]:
        self._ensure_today_stats()
        row = self._conn.execute(
            "SELECT ai_calls, images_posted, COALESCE(cost_rmb, 0) AS cost_rmb, "
            "COALESCE(support_calls, 0) AS support_calls "
            "FROM run_stats WHERE date = ?", (self._today(),)
        ).fetchone()
        return {"ai_calls": row["ai_calls"], "images_posted": row["images_posted"],
                "cost_rmb": row["cost_rmb"], "support_calls": row["support_calls"]}

    def overview(self) -> dict[str, int]:
        """Lifetime counts for `ic2x status` — posted / approved-pending / rejected
        image rows, plus the asset_index seen-set size. One cheap snapshot query set."""
        c = self._conn
        def _n(sql: str, *params: Any) -> int:
            row = c.execute(sql, params).fetchone()
            return row[0] if row else 0
        return {
            "posted": _n("SELECT COUNT(*) FROM images WHERE status = ?", Status.POSTED.value),
            "approved_pending": _n("SELECT COUNT(*) FROM images WHERE status = ?",
                                   Status.APPROVED.value),
            "rejected": _n("SELECT COUNT(*) FROM images WHERE status = ?", Status.REJECTED.value),
            "indexed": _n("SELECT COUNT(*) FROM asset_index"),
            "seen": _n("SELECT COUNT(*) FROM asset_index WHERE seen = 1"),
        }

    def recent_stats(self, days: int = 14) -> list[dict]:
        """Per-day (date, ai_calls, images_posted, cost_rmb), newest first — for `ic2x cost`."""
        rows = self._conn.execute(
            "SELECT date, ai_calls, images_posted, COALESCE(cost_rmb, 0) AS cost_rmb "
            "FROM run_stats ORDER BY date DESC LIMIT ?", (days,)
        ).fetchall()
        return [dict(r) for r in rows]

    def increment_color_calls(self) -> int:
        """Bump today's VIAPI color-enhance call count; return this calendar month's
        running total, so the caller can apply the free-N-per-month quota."""
        self._ensure_today_stats()
        self._conn.execute(
            "UPDATE run_stats SET color_calls = COALESCE(color_calls, 0) + 1 WHERE date = ?",
            (self._today(),),
        )
        self._conn.commit()
        return self.month_color_calls()

    def month_color_calls(self, month: str | None = None) -> int:
        """Total VIAPI color-enhance calls in `month` (YYYY-MM, default current month)."""
        month = month or self._today()[:7]
        row = self._conn.execute(
            "SELECT COALESCE(SUM(color_calls), 0) AS n FROM run_stats WHERE date LIKE ?",
            (f"{month}-%",),
        ).fetchone()
        return row["n"]

    def count_posts_rolling_24h(self) -> int:
        """Posts in the last 24h — a timezone-stable cap (the prod box is UTC+8)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        row = self._conn.execute(
            "SELECT COUNT(*) FROM images WHERE status = ? AND posted_at >= ?",
            (Status.POSTED.value, cutoff),
        ).fetchone()
        return row[0] if row else 0

    def recent_posted_phashes(self, limit: int) -> list[str]:
        """Phashes of the most recently POSTED images, newest first (scene dedup)."""
        rows = self._conn.execute(
            "SELECT phash FROM images WHERE status = ? AND phash IS NOT NULL "
            "ORDER BY posted_at DESC LIMIT ?",
            (Status.POSTED.value, limit),
        ).fetchall()
        return [r["phash"] for r in rows]

    def recent_captions(self, limit: int) -> list[str]:
        """First lines of the most recently POSTED captions, newest first.

        Fed to the caption pass so it can avoid reusing a construction it has just
        used. Without this the writer has no memory across posts, and the same shapes
        pile up: measured over 42 posts, 52% contained "here" and 38% "often", and an
        opener the prompt explicitly bans ("Westerners often …") still appeared twice.
        Only the text line is returned — the 📍 line is stripped."""
        rows = self._conn.execute(
            "SELECT caption FROM images WHERE status = ? AND caption IS NOT NULL "
            "AND caption != '' ORDER BY posted_at DESC LIMIT ?",
            (Status.POSTED.value, limit),
        ).fetchall()
        out = []
        for r in rows:
            first = (r["caption"] or "").split("\n")[0].strip()
            if first:
                out.append(first)
        return out

    # ── X reconciliation ───────────────────────────────────────────────────────

    def posted_for_reconcile(self, limit: int) -> list[sqlite3.Row]:
        """The most-recent POSTED rows with a real tweet_id (newest first) — the
        candidates whose existence on X is checked by id at startup."""
        return self._conn.execute(
            "SELECT id, asset_id, sha256, tweet_id, phash, caption, posted_at "
            "FROM images WHERE status = ? AND tweet_id IS NOT NULL AND tweet_id != 'DRYRUN' "
            "ORDER BY posted_at DESC LIMIT ?",
            (Status.POSTED.value, limit),
        ).fetchall()

    def requeue_deleted(self, image_id: int, asset_id: str | None) -> None:
        """Put a photo back in the pool: drop its images row and clear the iCloud asset
        from the seen-set, so the bot reconsiders and posts it again. Atomic.

        NOT what reconcile does with a post deleted on X — that retires it via
        reject_deleted(). This is the manual escape hatch for deliberately giving a
        retired photo another chance.
        """
        try:
            self._conn.execute("BEGIN")
            self._conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
            if asset_id:
                self._conn.execute(
                    "UPDATE asset_index SET seen = 0 WHERE asset_id = ?", (asset_id,))
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def reject_deleted(self, image_id: int, asset_id: str | None) -> None:
        """A post was deleted on X → record it as REJECTED and keep the asset in the
        seen-set, so the bot never posts that photo again. The owner deleting a post
        is a judgement about the photo, not a request to try it once more; re-queuing
        made the bot argue with them (2026-08-04: a post was deleted minutes after it
        went up and the photo went straight back into the pool). Atomic.

        Reversible by hand if a photo really should return:
            UPDATE images SET status='rejected' ...  -- find the row, then
            DELETE FROM images WHERE id=<id>; UPDATE asset_index SET seen=0 WHERE ...
        """
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                "UPDATE images SET status = ?, reject_stage = ?, reject_reason = ?, "
                "tweet_id = NULL WHERE id = ?",
                (Status.REJECTED.value, "deleted_on_x", "owner deleted the post on X",
                 image_id))
            if asset_id:
                self._conn.execute(
                    "INSERT INTO asset_index (asset_id, seen) VALUES (?, 1) "
                    "ON CONFLICT(asset_id) DO UPDATE SET seen = 1", (asset_id,))
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def update_caption(self, image_id: int, caption: str) -> None:
        """Fill/refresh a posted row's caption (used to sync from X)."""
        self._conn.execute("UPDATE images SET caption = ? WHERE id = ?", (caption, image_id))
        self._conn.commit()

    # ── Run state (generic key/value) ──────────────────────────────────────────

    # ── Capture-date catalog (chronological walk-back) ─────────────────────────

    def catalog_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM asset_catalog").fetchone()[0]

    def catalog_upsert_many(self, rows: list[tuple[str, str | None, int]]) -> None:
        """rows = (asset_id, created_iso | None, rank). Upsert keeps refreshes cheap."""
        self._conn.executemany(
            "INSERT INTO asset_catalog (asset_id, created, rank) VALUES (?,?,?) "
            "ON CONFLICT(asset_id) DO UPDATE SET created=excluded.created, rank=excluded.rank",
            rows)
        self._conn.commit()

    def catalog_known(self, asset_ids: list[str]) -> set[str]:
        """Subset of asset_ids already cataloged (for the head refresh)."""
        out: set[str] = set()
        for i in range(0, len(asset_ids), 500):
            chunk = asset_ids[i:i + 500]
            q = ",".join("?" * len(chunk))
            out.update(r[0] for r in self._conn.execute(
                f"SELECT asset_id FROM asset_catalog WHERE asset_id IN ({q})", chunk))
        return out

    def catalog_delete(self, asset_id: str) -> None:
        """Drop a catalog row whose asset was deleted from the iCloud library."""
        self._conn.execute("DELETE FROM asset_catalog WHERE asset_id = ?", (asset_id,))
        self._conn.commit()

    def catalog_unseen_desc(self) -> list:
        """(asset_id, created, rank) newest-capture-first, excluding assets the bot
        has already decided on — the chronological walk-back order. ~4 MB for a
        67k-photo library; loaded once per cycle. The scanner re-checks seen
        downstream, so mid-cycle staleness is harmless."""
        return self._conn.execute(
            "SELECT c.asset_id, c.created, c.rank FROM asset_catalog c "
            "LEFT JOIN asset_index i ON i.asset_id = c.asset_id AND i.seen = 1 "
            "WHERE i.asset_id IS NULL AND c.created IS NOT NULL "
            "ORDER BY c.created DESC, c.asset_id").fetchall()

    def get_state(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM run_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row and row["value"] is not None else None

    def set_state(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO run_state (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    def get_last_posted_at(self) -> datetime | None:
        raw = self.get_state("last_posted_at")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def set_last_posted_at(self, dt: datetime) -> None:
        self.set_state("last_posted_at", dt.isoformat())

    def refresh_last_posted_at(self) -> None:
        """Recompute last_posted_at from the most recent STILL-LIVE post, so a post the
        owner deleted on X no longer holds the posting interval. Clears it (→ due now) if
        nothing real is posted. Called by reconcile after it re-queues a deletion."""
        row = self._conn.execute(
            "SELECT MAX(posted_at) AS m FROM images "
            "WHERE status = ? AND tweet_id IS NOT NULL AND tweet_id != 'DRYRUN'",
            (Status.POSTED.value,),
        ).fetchone()
        self.set_state("last_posted_at", (row["m"] if row else None) or "")

    # ── Clean ──────────────────────────────────────────────────────────────────

    _CLEANABLE_STATUSES = (Status.QUEUED, Status.APPROVED, Status.SEEN)

    def get_cleanable_counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for status in self._CLEANABLE_STATUSES:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM images WHERE status = ?", (status.value,)
            ).fetchone()
            result[status.value] = row[0]
        return result

    def clean_pipeline(self) -> int:
        """Delete non-posted image records (keeps posted history). Returns count."""
        ph = ",".join("?" * len(self._CLEANABLE_STATUSES))
        cur = self._conn.execute(
            f"DELETE FROM images WHERE status IN ({ph})",
            [s.value for s in self._CLEANABLE_STATUSES],
        )
        self._conn.commit()
        return cur.rowcount
