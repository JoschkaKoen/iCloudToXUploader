"""
Offline tests for X weighted-length helpers (ic2x/utils/tweet_text.py).

Proves emoji/CJK weigh 2, ASCII weighs 1, truncation respects the weighted budget
and never splits a code point, and the real bug case — a caption that is ≤280
Python chars but >280 X-weighted — is trimmed so a post can't 403 on length.

Run: .venv/bin/python tests/test_tweet_text.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x.utils.tweet_text import truncate_weighted, weighted_len  # noqa: E402


def test_ascii_weighs_one():
    assert weighted_len("hello world") == len("hello world")
    assert weighted_len("") == 0


def test_emoji_weighs_two():
    assert weighted_len("📍") == 2          # one code point, weight 2
    assert weighted_len("🍜🌅") == 4
    assert weighted_len("a📍b") == 4         # 1 + 2 + 1


def test_cjk_weighs_two():
    assert weighted_len("中国") == 4
    assert weighted_len("Ningbo 中国") == len("Ningbo ") + 4


def test_location_line_weight():
    loc = "📍 19:50 Ningbo, China"
    # exactly one heavy char (📍): weight = python-len + 1
    assert weighted_len(loc) == len(loc) + 1


def test_truncate_respects_weighted_budget():
    s = "x" * 10 + "📍" * 10          # 10 + 20 = 30 weighted, 20 code points
    out = truncate_weighted(s, 14)    # 10 'x' (10) + 2 emoji (4) = 14
    assert weighted_len(out) <= 14
    assert out == "x" * 10 + "📍" * 2


def test_truncate_never_splits_a_codepoint():
    # budget lands mid-emoji → the emoji is dropped whole, not half-written
    out = truncate_weighted("ab📍", 3)   # a(1)+b(1)=2, emoji would push to 4 > 3
    assert out == "ab"
    assert weighted_len(out) <= 3


def test_truncate_zero_or_negative():
    assert truncate_weighted("hello", 0) == ""
    assert truncate_weighted("hello", -5) == ""


def test_real_bug_case_caption_with_emoji_over_280():
    # 279 ASCII + one emoji = 281 weighted but only 280 python chars: old len()-based
    # cap would have let this through to X and 403'd.
    caption = "a" * 279 + "🌅"
    assert len(caption) == 280 and weighted_len(caption) == 281
    out = truncate_weighted(caption, 280)
    assert weighted_len(out) <= 280
    # the trailing emoji (weight 2) can't fit in the last slot, so it's dropped
    assert out == "a" * 279


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback; print(f"FAIL {t.__name__}: {exc}"); traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())


# ── build_tweet: caption + 📍 line + at most ONE hashtag ────────────────────────

def test_build_tweet_orders_and_keeps_one_tag():
    from ic2x.utils.tweet_text import build_tweet
    loc = "📍 18:04 Shanghai, China"
    out = build_tweet("Luxury flagships sit by subway entrances. 🛍", loc, "#China")
    assert out.split("\n") == ["Luxury flagships sit by subway entrances. 🛍", loc, "#China"]
    # a bare word is normalised, and a multi-word value never becomes two tags
    assert build_tweet("x", None, "China").endswith("#China")
    assert build_tweet("x", None, "#China #Travel").count("#") == 1


def test_build_tweet_trims_caption_never_location_or_tag():
    """The 📍 line and the tag are fixed costs reserved up front — a long caption must
    never push either off the end or blow the 280 weighted budget (a 403 at post time)."""
    from ic2x.utils.tweet_text import build_tweet, weighted_len
    loc = "📍 18:04 Shanghai, China"
    out = build_tweet("word " * 200, loc, "#China")
    assert weighted_len(out) <= 280
    assert out.endswith("#China") and loc in out


def test_build_tweet_untagged_arm_and_no_location():
    from ic2x.utils.tweet_text import build_tweet
    loc = "📍 18:04 Shanghai, China"
    assert build_tweet("A short one. 🍜", loc, "") == f"A short one. 🍜\n{loc}"   # control arm
    assert build_tweet("No GPS. 🏙", None, "#China") == "No GPS. 🏙\n#China"
    assert build_tweet("", loc, "") == loc          # no leading blank line
