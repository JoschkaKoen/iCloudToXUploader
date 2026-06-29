"""
Single source of truth for the image state machine.

Pipeline transitions (the best-of-burst bot):

    seen → approved → posting → posted
                             ↘ rejected (from any earlier stage)

`QUEUED` is a legacy status from the old multi-stage pipeline — the current loop
prepares the winner straight to `approved`, so it never sets `QUEUED`. It is kept
in the enum because `ic2x clean` still treats any leftover `queued` row as
discardable.

`posting` is the idempotency anchor — written before the tweet API call,
then advanced to `posted` after. Any row stuck in `posting` means the process
was killed mid-call; the bot auto-recovers it to `approved` at the start of
each cycle via `DB.reset_stuck_posting()`, so `flush_pending` retries it.

Usage notes
-----------
- `Status` inherits from `str`, so `Status.SEEN == "seen"` is True and the
  enum binds directly as a SQL parameter:
      conn.execute("UPDATE images SET status = ? WHERE …", (Status.SEEN, …))
- Inside Python f-strings the enum stringifies to `"Status.SEEN"`, NOT to
  `"seen"` (Python 3.11+). Use `Status.SEEN.value` for any literal text:
      f"… WHERE status = '{Status.SEEN.value}' …"   # SQL literal
      logger.info("status=%s", Status.SEEN.value)   # log message
"""

from __future__ import annotations

from enum import Enum


class Status(str, Enum):
    SEEN     = "seen"
    QUEUED   = "queued"
    APPROVED = "approved"
    POSTING  = "posting"
    POSTED   = "posted"
    REJECTED = "rejected"
