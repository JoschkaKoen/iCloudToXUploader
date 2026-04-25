"""
Single source of truth for the image state machine.

Pipeline transitions:

    seen → queued → approved → posting → posted
                                       ↘ rejected (from any earlier stage)

`posting` is the idempotency anchor for `ic2x post` — written before the
tweet API call, then advanced to `posted` after. Any row stuck in `posting`
on startup means the process was killed mid-call and must be reconciled
before retrying (see `ic2x unstick`).

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
