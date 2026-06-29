"""
Shared pytest setup.

Some tests build a real ``Config`` via ``load_config()`` (e.g. test_bot_testmode),
which requires the iCloud + X credential env vars to be present. Tests never make
real network calls — the iCloud/X/AI clients are monkeypatched — so dummy values
are all that's needed. We set them here (before any test module imports
``ic2x.config``) so the whole suite runs under ``pytest`` with no ``.env`` file.

``setdefault`` means a real value in the environment (or a loaded ``.env``) always
wins, so this never masks a developer's actual credentials.
"""

from __future__ import annotations

import os

_DUMMY_ENV = {
    "ICLOUD_USERNAME": "test@example.com",
    "ICLOUD_PASSWORD": "test-password",
    "TWITTER_CONSUMER_KEY": "test-ck",
    "TWITTER_CONSUMER_SECRET": "test-cs",
    "TWITTER_ACCESS_TOKEN": "test-at",
    "TWITTER_ACCESS_TOKEN_SECRET": "test-ats",
}

for _k, _v in _DUMMY_ENV.items():
    os.environ.setdefault(_k, _v)
