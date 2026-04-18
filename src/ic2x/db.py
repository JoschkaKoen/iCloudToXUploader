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
    gemini_calls    INTEGER DEFAULT 0,
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

    def seen_phash_similar(self, phash: str, threshold: int) -> bool:
        import imagehash
        target = imagehash.hex_to_hash(phash)
        rows = self._conn.execute(
            "SELECT phash FROM images WHERE status = 'posted' AND phash IS NOT NULL"
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

    def get_approved(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM images WHERE status = 'approved'"
        ).fetchall()

    # ── Gemini call budget ────────────────────────────────────────────────────

    def _today(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d")

    def _ensure_today_stats(self) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO run_stats (date) VALUES (?)", (self._today(),)
        )
        self._conn.commit()

    def check_daily_gemini_limit(self, limit: int) -> bool:
        """Return True if the daily limit has been reached."""
        self._ensure_today_stats()
        row = self._conn.execute(
            "SELECT gemini_calls FROM run_stats WHERE date = ?", (self._today(),)
        ).fetchone()
        return (row["gemini_calls"] if row else 0) >= limit

    def increment_gemini_calls(self, n: int = 1) -> None:
        self._ensure_today_stats()
        self._conn.execute(
            "UPDATE run_stats SET gemini_calls = gemini_calls + ? WHERE date = ?",
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
