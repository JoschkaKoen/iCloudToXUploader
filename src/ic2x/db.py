"""
All SQLite access lives here. Nothing else touches the DB directly.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
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
    processed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    posted_at       TIMESTAMP
);

CREATE TABLE IF NOT EXISTS run_stats (
    date            TEXT PRIMARY KEY,
    ai_calls        INTEGER DEFAULT 0,
    images_pulled   INTEGER DEFAULT 0,
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
        # One-time migration: rename legacy gemini_calls column to ai_calls.
        # Safe on both fresh databases (table doesn't exist yet) and existing ones
        # (column already renamed → silently ignored).
        try:
            self._conn.execute(
                "ALTER TABLE run_stats RENAME COLUMN gemini_calls TO ai_calls"
            )
            self._conn.commit()
        except Exception:
            pass
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Image lookups ─────────────────────────────────────────────────────────

    def seen_sha256(self, sha256: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM images WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return row is not None

    def filenames_processed(self, filenames: set[str]) -> set[str]:
        """Return the subset of filenames that already have a completed DB record.

        'Completed' means status beyond 'seen' (queued, approved, posted, rejected).
        Used by pull.py to skip already-processed files before any pipeline step.
        """
        if not filenames:
            return set()
        placeholders = ",".join("?" * len(filenames))
        rows = self._conn.execute(
            f"SELECT source_filename FROM images "
            f"WHERE source_filename IN ({placeholders}) AND status != 'seen'",
            list(filenames),
        ).fetchall()
        return {row["source_filename"] for row in rows}

    def seen_phash_similar(self, phash: str, threshold: int) -> bool:
        import imagehash
        target = imagehash.hex_to_hash(phash)
        rows = self._conn.execute(
            "SELECT phash FROM images WHERE status NOT IN ('rejected') AND phash IS NOT NULL"
        ).fetchall()
        for row in rows:
            try:
                candidate = imagehash.hex_to_hash(row["phash"])
                if (target - candidate) <= threshold:
                    return True
            except Exception:
                continue
        return False

    def insert_seen(self, sha256: str, phash: str, filename: str) -> int:
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO images (sha256, phash, source_filename, status) "
            "VALUES (?, ?, ?, 'seen')",
            (sha256, phash, filename),
        )
        self._conn.commit()
        return cur.lastrowid

    def set_status(self, sha256: str, status: str, **kwargs: Any) -> None:
        allowed = {
            "reject_stage", "reject_reason", "safety_raw", "quality_raw",
            "caption", "tweet_id", "posted_at", "phash",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        # Serialise dicts/lists to JSON
        for k, v in updates.items():
            if isinstance(v, (dict, list)):
                updates[k] = json.dumps(v)

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values())
        if set_clause:
            self._conn.execute(
                f"UPDATE images SET status = ?, {set_clause} WHERE sha256 = ?",
                [status] + values + [sha256],
            )
        else:
            self._conn.execute(
                "UPDATE images SET status = ? WHERE sha256 = ?",
                (status, sha256),
            )
        self._conn.commit()

    def get_stuck_posting(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM images WHERE status = 'posting'"
        ).fetchall()

    def reset_stuck_posting(self) -> int:
        """Reset all rows stuck in 'posting' back to 'approved'. Returns count updated."""
        cur = self._conn.execute(
            "UPDATE images SET status='approved' WHERE status='posting'"
        )
        self._conn.commit()
        return cur.rowcount

    def get_approved(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM images WHERE status = 'approved'"
        ).fetchall()

    # ── AI call budget ────────────────────────────────────────────────────────

    def _today(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d")

    def _ensure_today_stats(self) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO run_stats (date) VALUES (?)", (self._today(),)
        )
        self._conn.commit()

    def check_daily_ai_limit(self, limit: int) -> bool:
        """Return True if the daily AI call limit has been reached."""
        self._ensure_today_stats()
        row = self._conn.execute(
            "SELECT ai_calls FROM run_stats WHERE date = ?", (self._today(),)
        ).fetchone()
        return (row["ai_calls"] if row else 0) >= limit

    def increment_ai_calls(self, n: int = 1) -> None:
        self._ensure_today_stats()
        self._conn.execute(
            "UPDATE run_stats SET ai_calls = ai_calls + ? WHERE date = ?",
            (n, self._today()),
        )
        self._conn.commit()

    def increment_images_pulled(self, n: int = 1) -> None:
        self._ensure_today_stats()
        self._conn.execute(
            "UPDATE run_stats SET images_pulled = images_pulled + ? WHERE date = ?",
            (n, self._today()),
        )
        self._conn.commit()

    def increment_images_posted(self, n: int = 1) -> None:
        self._ensure_today_stats()
        self._conn.execute(
            "UPDATE run_stats SET images_posted = images_posted + ? WHERE date = ?",
            (n, self._today()),
        )
        self._conn.commit()

    # ── Run state ─────────────────────────────────────────────────────────────

    def get_last_asset_date(self) -> datetime | None:
        row = self._conn.execute(
            "SELECT value FROM run_state WHERE key = 'last_asset_date'"
        ).fetchone()
        if not row or not row["value"]:
            return None
        try:
            return datetime.fromisoformat(row["value"])
        except ValueError:
            return None

    def set_last_asset_date(self, dt: datetime) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO run_state (key, value) VALUES ('last_asset_date', ?)",
            (dt.isoformat(),),
        )
        self._conn.commit()

    def get_last_posted_at(self) -> datetime | None:
        row = self._conn.execute(
            "SELECT value FROM run_state WHERE key = 'last_posted_at'"
        ).fetchone()
        if not row or not row["value"]:
            return None
        try:
            return datetime.fromisoformat(row["value"])
        except ValueError:
            return None

    def set_last_posted_at(self, dt: datetime) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO run_state (key, value) VALUES ('last_posted_at', ?)",
            (dt.isoformat(),),
        )
        self._conn.commit()

    def get_lookback_depth(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM run_state WHERE key = 'icloud_lookback_depth'"
        ).fetchone()
        if not row or not row["value"]:
            return 0
        try:
            return int(row["value"])
        except (ValueError, TypeError):
            return 0

    def set_lookback_depth(self, n: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO run_state (key, value) VALUES ('icloud_lookback_depth', ?)",
            (str(n),),
        )
        self._conn.commit()

    # ── Image lookup by SHA ───────────────────────────────────────────────────

    def get_image_by_sha(self, sha256: str) -> "sqlite3.Row | None":
        return self._conn.execute(
            "SELECT * FROM images WHERE sha256 = ?", (sha256,)
        ).fetchone()
