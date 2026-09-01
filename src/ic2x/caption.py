"""
Winner caption pass — one VLM call that captions the chosen photo, grounded in WHERE and
WHEN it was taken.

The burst judge only sees EXIF-stripped thumbnails, so it can't know the city. This pass
runs after the winner's original is downloaded — once the reverse-geocoded city and local
time are known — and is told both in a single prompt. Best-effort: any failure returns
(None, used) and the caller keeps the judge's caption. The 📍 line is added by the caller.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ic2x.config import Config
from ic2x.utils.ai_client import (
    JudgeCall,
    call_vision_judge,
    parse_model_effort,
    provider_for_model,
)

logger = logging.getLogger("ic2x.caption")

_RECENT_MAX = 8     # enough shape history to break a rut; a few hundred input tokens


def caption_model(cfg: Config) -> str:
    """Model that WRITES the caption. Defaults to the judge model, but is its own knob
    because judging a photo and writing about it are different jobs — benchmarked
    2026-08-09, every Grok tier was cheaper than qwen3.7-plus (grok-4-1-fast at 0.12x)
    yet described the photo instead of drawing an insight from it, so qwen stays the
    default. The knob makes re-testing that a config change, not a code change."""
    return getattr(cfg, "caption_model", "") or cfg.judge_model

# One prompt; the photo's place and time are interpolated in. {{ }} are literal JSON braces.
_CAPTION_PROMPT = """You are a long-time expat in China, captioning your own photo on X (twitter) for
Western viewers who have never visited China.
The photo you are seeing was taken in {place} at {when}.

Write ONE short sentence, UNDER 120 characters. Shorter is better — every word
must earn its place.
Share a real, specific insight a Western viewer can learn from it.
ACCURACY: call things what they are; never invent facts, meanings or prices.
Name or translate text ONLY when it is CLEARLY readable and you are certain —
never guess from logo shape/colors or small/distant/blurry text; when in doubt,
leave names and translations out.
Translate readable Chinese only for everyday COMMERCIAL content (shops, menus,
stalls). NEVER translate, quote or comment on government, civic or official
messaging — describe such scenes generally instead.
TONE: plain and matter-of-fact, like a knowledgeable friend texting — never
promotional or awed. Banned words: vibrant, bustling, stunning, lush,
incredible, amazing, breathtaking. Never negative either.
Two crutches to avoid. Never frame it around what outsiders believe ("Many
assume", "Westerners think", any variant) — state the fact directly. And never
hedge a pattern with "often/typically/usually/commonly": present tense is already
general — "Bakeries top fried dough with cream", not "Bakeries here often top".
I LIVE IN CHINA — never critical, political or mocking; no "slogan"/
"propaganda"-type framing.
{recent_block}
No hashtags, no emoji inside the text.
Never restate the photo's location or time (appended automatically).

Return ONLY JSON: {{"caption": "<caption>", "emoji": "<REQUIRED: one emoji for the \
MAIN subject, not background details>"}}"""


_RECENT_BLOCK = """
My recent captions — do NOT reuse their sentence shape, nor the generalising crutch
they lean on ("<thing> here often …"):
{recent}
Vary the opening, but stay SHORT and still teach something: a fresh shape that only
describes what is visible is worse than none.
"""


_PICKER_PROMPT_HEAD = """You are choosing the best of {n} candidate tweets for this photo. The tweet is
written by an expat living in China, for Western viewers who have never visited China.

The writer LIVES IN CHINA — the tweet must never cause them problems.

Pick the candidate that best satisfies, in this order:
(1) SAFE for a China resident: nothing critical, political or mocking; never quotes,
    translates or comments on government/civic/official messaging (a candidate doing
    so is DISQUALIFIED — prefer one describing the scene generally);
(2) factually ACCURATE about what the photo shows — a mislabeled object or an
    invented/unverifiable claim disqualifies; naming a specific building or place
    whose name is NOT clearly readable in the photo counts as invented; a brand
    name or translation that appears in only ONE candidate was probably misread —
    prefer candidates without it;
(3) translates everyday COMMERCIAL Chinese text (shops, menus, stalls) when prominent;
(4) teaches a Western viewer something real about China;
(5) plain, matter-of-fact tone — promotional or awed wording ("vibrant", "stunning",
    "lush", "incredible") DISQUALIFIES;
(6) is SHORT — among candidates equal on the above, always prefer the shortest;
(7) ends with one emoji matching the photo's MAIN subject.

Candidates:
"""

_PICKER_PROMPT_TAIL = """
Return ONLY valid JSON — no markdown:
{"best_index": <int>, "reason": "<max 15 words>"}"""


def generate_caption(image_path: Path, place: str | None, when: str | None,
                     cfg: Config, recent: list[str] | None = None) -> tuple[str | None, int]:
    """Caption the winner image, grounded in `place` + `when`.

    Best-of-N (cfg.caption_candidates, default 1): N independent caption calls run in
    parallel, then one picker call chooses the best tweet (see _pick_best). Returns
    (caption, n_network_calls); caption is None on total failure (caller keeps the
    judge's caption). Skipped on a local-only/ollama model.

    `recent` is the last few posted captions. The writer is otherwise stateless across
    posts, so it cannot tell it is reusing a shape it just used — over 42 posts 52%
    contained "here" and 38% "often". Showing it the recent ones costs a few hundred
    input tokens and is the only thing that can break that, since the picker only ever
    sees candidates for the current photo."""
    model, _ = parse_model_effort(caption_model(cfg))
    if provider_for_model(model) == "ollama":
        return None, 0  # the caption pass needs a cloud VLM

    recent_block = ""
    if recent:
        listed = "\n".join(f"- {c}" for c in recent[:_RECENT_MAX])
        recent_block = _RECENT_BLOCK.format(recent=listed)
    prompt = _CAPTION_PROMPT.format(place=place or "an unknown place",
                                    when=when or "an unknown time",
                                    recent_block=recent_block)
    n = max(1, int(getattr(cfg, "caption_candidates", 1) or 1))
    if n == 1:
        cap, used = _generate_one(image_path, prompt, cfg)
        return cap, int(used)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda _i: _generate_one(image_path, prompt, cfg), range(n)))
    used_count = sum(int(u) for _c, u in results)

    seen: set[str] = set()
    cands: list[str] = []
    for c, _u in results:
        if c and c not in seen:
            seen.add(c)
            cands.append(c)
    if not cands:
        return None, used_count
    if len(cands) == 1:
        return cands[0], used_count

    idx, pick_used = _pick_best(image_path, cands, cfg)
    used_count += int(pick_used)
    logger.info("caption: best-of-%d picked candidate %d/%d", n, idx, len(cands))
    winner = cands[idx]
    if not _is_emoji_char(winner[-1]):
        # accuracy can outrank emoji at pick time — borrow the consensus emoji
        # from the other candidates (no extra call) so the account style holds
        from collections import Counter
        pool = Counter(c[-1] for c in cands if c is not winner and _is_emoji_char(c[-1]))
        if pool:
            winner = f"{winner} {pool.most_common(1)[0][0]}"
    return winner, used_count


def _generate_one(image_path: Path, prompt: str, cfg: Config) -> tuple[str | None, bool]:
    """One caption call. Returns (caption|None, used_network)."""
    parsed, _elapsed, ok, used = call_vision_judge(
        model_string=caption_model(cfg),
        ollama_base_url=cfg.ollama_base_url,
        call=JudgeCall(
            image_path=image_path,
            prompt=prompt,
            max_px=cfg.judge_image_max_px,
            fail_value={"caption": ""},
            refused_value={"caption": ""},
            label="caption",
        ),
    )
    if not ok:
        return None, used

    caption = " ".join((parsed.get("caption") or "").split())  # normalise whitespace
    if caption:
        caption = _append_subject_emoji(caption, parsed.get("emoji"))
    return (caption or None), used


def _pick_best(image_path: Path, candidates: list[str], cfg: Config) -> tuple[int, bool]:
    """One picker call: the photo + numbered candidates → best_index.
    Returns (index, used_network); falls back to index 0 on any failure."""
    numbered = "\n".join(f"{i}: {c}" for i, c in enumerate(candidates))
    prompt = (_PICKER_PROMPT_HEAD.format(n=len(candidates)) + numbered
              + _PICKER_PROMPT_TAIL)
    model_str = getattr(cfg, "caption_picker_model", "") or caption_model(cfg)
    parsed, _elapsed, ok, used = call_vision_judge(
        model_string=model_str,
        ollama_base_url=cfg.ollama_base_url,
        call=JudgeCall(
            image_path=image_path,
            prompt=prompt,
            max_px=cfg.judge_image_max_px,
            fail_value={"best_index": 0},
            refused_value={"best_index": 0},
            label="caption_pick",
        ),
    )
    if not ok:
        return 0, used
    try:
        idx = int(parsed.get("best_index", 0))
    except (TypeError, ValueError):
        idx = 0
    if not 0 <= idx < len(candidates):
        idx = 0
    return idx, used


def _is_emoji_char(c: str) -> bool:
    """Emoji/symbol planes; CJK text chars excluded."""
    o = ord(c)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF


def _append_subject_emoji(caption: str, emoji) -> str:
    """Append the model's dedicated subject-emoji field. The emoji lives in its own
    JSON field (not the prose) because prompt-only "end with an emoji" instructions
    proved flaky — models dropped it in ~1/3 of samples. The field is scanned for its
    first emoji char (models sometimes pad it with words); never doubles up if the
    caption already ends with one."""
    e = ""
    if isinstance(emoji, str):
        e = next((c for c in emoji if _is_emoji_char(c)), "")
    if not e:
        return caption
    if caption and _is_emoji_char(caption[-1]):
        return caption
    return f"{caption} {e}"
