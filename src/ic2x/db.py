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
CREATE TABLE IF NOT EXISTS asset_index (
    asset_id      TEXT PRIMARY KEY,
    created       TEXT NOT NULL,          -- ISO8601 UTC capture date
    filename      TEXT,
    is_screenshot INTEGER DEFAULT 0,
    is_live       INTEGER DEFAULT 0,
    seen          INTEGER DEFAULT 0,      -- 1 once the bot has decided on it
    attempts      INTEGER DEFAULT 0,      -- pre-commit failures (poison-burst breaker)
    width         INTEGER,
    height        INTEGER,
    indexed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_asset_created ON asset_index(created);
CREATE INDEX IF NOT EXISTS idx_asset_seen ON asset_index(seen);

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
    ai_calls        INTEGER DEFAULT 0,
    images_posted   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS run_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class DB:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Asset index (synced from iCloud metadata) ──────────────────────────────

    def upsert_asset(
        self, asset_id: str, created: datetime | str, filename: str,
        *, is_live: bool = False, width: int | None = None, height: int | None = None,
    ) -> bool:
        """Insert a still-image asset if new. Returns True if it was new.
        Immutable fields (created/filename/dims) are not overwritten on resync."""
        created_iso = created.isoformat() if isinstance(created, datetime) else str(created)
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO asset_index "
            "(asset_id, created, filename, is_live, width, height) VALUES (?, ?, ?, ?, ?, ?)",
            (asset_id, created_iso, filename, int(is_live), width, height),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def asset_indexed(self, asset_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM asset_index WHERE asset_id = ?", (asset_id,)
        ).fetchone() is not None

    def asset_index_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM asset_index").fetchone()[0]

    def mark_screenshots(self, asset_ids: Sequence[str]) -> int:
        """Flag the given asset_ids as screenshots (Apple's Screenshots album)."""
        if not asset_ids:
            return 0
        ph = ",".join("?" * len(asset_ids))
        cur = self._conn.execute(
            f"UPDATE asset_index SET is_screenshot = 1 WHERE asset_id IN ({ph})",
            list(asset_ids),
        )
        self._conn.commit()
        return cur.rowcount

    def next_unseen_assets(self, limit: int) -> list[sqlite3.Row]:
        """The newest `limit` not-yet-decided assets, newest capture first.
        Drives 'prefer new, else walk back' purely via the `seen` flag."""
        return self._conn.execute(
            "SELECT * FROM asset_index WHERE seen = 0 ORDER BY created DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def has_unseen_assets(self) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM asset_index WHERE seen = 0 LIMIT 1"
        ).fetchone() is not None

    def incr_asset_attempts(self, asset_id: str) -> int:
        """Bump and return the pre-commit attempt count for a burst-head asset."""
        self._conn.execute(
            "UPDATE asset_index SET attempts = attempts + 1 WHERE asset_id = ?", (asset_id,)
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
        """Atomically mark every burst member decided, and (optionally) insert the
        winner's images row. A crash before this leaves zero state → the identical
        burst re-assembles cleanly; a crash after → flush_pending finishes the post.

        winner keys: asset_id, sha256, phash, filename, status (Status|str),
        caption, reject_stage, reject_reason.
        """
        try:
            self._conn.execute("BEGIN")
            if seen_asset_ids:
                ph = ",".join("?" * len(seen_asset_ids))
                self._conn.execute(
                    f"UPDATE asset_index SET seen = 1 WHERE asset_id IN ({ph})",
                    list(seen_asset_ids),
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

    def increment_ai_calls(self, n: int = 1) -> None:
        self._ensure_today_stats()
        self._conn.execute(
            "UPDATE run_stats SET ai_calls = ai_calls + ? WHERE date = ?", (n, self._today())
        )
        self._conn.commit()

    def increment_images_posted(self, n: int = 1) -> None:
        self._ensure_today_stats()
        self._conn.execute(
            "UPDATE run_stats SET images_posted = images_posted + ? WHERE date = ?",
            (n, self._today()),
        )
        self._conn.commit()

    def count_posts_rolling_24h(self) -> int:
        """Posts in the last 24h — a timezone-stable cap (the prod box is UTC+8)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        row = self._conn.execute(
            "SELECT COUNT(*) FROM images WHERE status = ? AND posted_at >= ?",
            (Status.POSTED.value, cutoff),
        ).fetchone()
        return row[0] if row else 0

    # ── Run state (generic key/value) ──────────────────────────────────────────

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
