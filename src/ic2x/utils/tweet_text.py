"""
X (Twitter) post-length helpers.

X does NOT count characters by Python ``len()``: under the twitter-text weighted
model most characters weigh 1, but CJK ideographs, kana, Hangul, fullwidth forms
and (crucially here) emoji weigh **2**. A caption that is ≤280 Python chars but
carries a couple of emoji + the "📍 …" line can therefore exceed X's 280-weight
limit and get rejected with a 403 at post time.

``weighted_len`` implements the twitter-text v3 default weighting for the ranges
that actually occur in this bot's captions (English text + emoji + a geocoded,
English city line). Emoji are approximated as "any code point in an emoji block =
2"; this can slightly over-count exotic ZWJ emoji sequences, which only ever makes
the result *shorter* (safe — never causes a 403). ``truncate_weighted`` trims a
string to a weighted budget on code-point boundaries.
"""

from __future__ import annotations

# Weight-2 code-point ranges (inclusive). The first block is the twitter-text v3
# default heavy ranges (CJK / kana / Hangul / fullwidth); the rest cover emoji.
_HEAVY_RANGES: tuple[tuple[int, int], ...] = (
    (0x1100, 0x115F),    # Hangul Jamo
    (0x2E80, 0x303E),    # CJK radicals, Kangxi, CJK symbols
    (0x3041, 0x33FF),    # Hiragana/Katakana, CJK symbols & punctuation
    (0x3400, 0x4DBF),    # CJK Ext-A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0xA000, 0xA4CF),    # Yi
    (0xA960, 0xA97F),    # Hangul Jamo Extended-A
    (0xAC00, 0xD7A3),    # Hangul Syllables
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0xFE30, 0xFE4F),    # CJK Compatibility Forms
    (0xFF00, 0xFF60),    # Fullwidth forms
    (0xFFE0, 0xFFE6),    # Fullwidth signs
    # Emoji / pictographic blocks (X counts these as 2):
    (0x2190, 0x21FF),    # arrows (emoji-presented)
    (0x2300, 0x23FF),    # misc technical (⌚⏰…)
    (0x2460, 0x24FF),    # enclosed alphanumerics (often emoji-presented)
    (0x25A0, 0x27BF),    # geometric shapes, misc symbols, dingbats (☀✨…)
    (0x2B00, 0x2BFF),    # misc symbols & arrows (⭐…)
    (0x1F000, 0x1FAFF),  # the main emoji planes (📍🍜🌅🏞…)
    (0x1FB00, 0x1FBFF),  # symbols for legacy computing
)

# Zero-width joiners / variation selectors / skin-tone modifiers contribute 0 to a
# tweet's weight (they only modify the preceding emoji, already counted as 2).
_ZERO_WIDTH: frozenset[int] = frozenset(
    {0x200D, 0xFE0E, 0xFE0F} | set(range(0x1F3FB, 0x1F400))  # ZWJ, VS-15/16, skin tones
)


def _char_weight(cp: int) -> int:
    if cp in _ZERO_WIDTH:
        return 0
    for lo, hi in _HEAVY_RANGES:
        if lo <= cp <= hi:
            return 2
    return 1


def weighted_len(text: str) -> int:
    """X-weighted length of *text* (most chars 1; CJK/fullwidth/emoji 2)."""
    return sum(_char_weight(ord(ch)) for ch in text)


def truncate_weighted(text: str, limit: int) -> str:
    """Longest prefix of *text* whose :func:`weighted_len` is ≤ ``limit``
    (code-point boundaries; never splits a code point)."""
    if limit <= 0:
        return ""
    total = 0
    out: list[str] = []
    for ch in text:
        w = _char_weight(ord(ch))
        if total + w > limit:
            break
        total += w
        out.append(ch)
    return "".join(out)
