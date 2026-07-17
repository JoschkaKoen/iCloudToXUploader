"""
Offline tests for the winner caption pass + the geo place/time split.

No network — call_vision_judge is monkeypatched. Proves the 📍 line formats from
(place, time); the single prompt interpolates place + time; and generate_caption returns
the model's caption, caps length cleanly, fails safe to None, and skips on a local model.

Run: .venv/bin/python tests/test_caption.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ic2x.caption as cap  # noqa: E402
from ic2x import geo  # noqa: E402


def _cfg():
    return SimpleNamespace(judge_model="qwen3.7-plus",
                           ollama_base_url="http://localhost:11434/v1",
                           judge_image_max_px=1024)


def test_format_location_line():
    assert geo.format_location_line("Ningbo, China", "19:50") == "📍 19:50 Ningbo, China"
    assert geo.format_location_line("Ningbo, China", None) == "📍 Ningbo, China"
    assert geo.format_location_line(None, "19:50") is None
    assert geo.format_location_line(None, None) is None


def test_prompt_interpolates_place_time_and_stays_concise():
    p = cap._CAPTION_PROMPT.format(place="Ningbo, China", when="19:50")
    assert "Ningbo, China" in p and "19:50" in p   # place + time reach the model
    # Focused template, not a rule pile — but every rule here maps to a REAL posted
    # caption the owner deleted (politics/tone/length/accuracy steers of 2026-07-12/13),
    # so the ceiling was consciously raised from 1200 once those were encoded.
    assert len(cap._CAPTION_PROMPT) < 1400


def _patch(result):
    """Patch the vision call + provider check; return (captured, restore)."""
    captured = {}
    orig = (cap.call_vision_judge, cap.provider_for_model, cap.parse_model_effort)

    def fake_call(*, model_string, ollama_base_url, call):
        captured["prompt"] = call.prompt
        captured["label"] = call.label
        return result

    cap.call_vision_judge = fake_call
    cap.provider_for_model = lambda m: "dashscope"
    cap.parse_model_effort = lambda s: (s, None)

    def restore():
        cap.call_vision_judge, cap.provider_for_model, cap.parse_model_effort = orig

    return captured, restore


def test_returns_model_caption_and_grounds_prompt():
    text = "Locals walk barefoot on these pebble paths for foot reflexology 🦶"
    captured, restore = _patch(({"caption": text}, 0.2, True, True))
    try:
        out, used = cap.generate_caption(Path("/x.jpg"), "Ningbo, China", "19:50", _cfg())
    finally:
        restore()
    assert out == text and used == 1
    assert "Ningbo, China" in captured["prompt"] and "19:50" in captured["prompt"]
    assert captured["label"] == "caption"


def test_no_gps_renders_cleanly():
    captured, restore = _patch(({"caption": "a tidy lane 🌿"}, 0.1, True, True))
    try:
        cap.generate_caption(Path("/x.jpg"), None, None, _cfg())
    finally:
        restore()
    assert "an unknown place" in captured["prompt"] and "an unknown time" in captured["prompt"]


def test_caption_whitespace_normalised_not_truncated():
    captured, restore = _patch(({"caption": "line one\n  line two   spaced"}, 0.1, True, True))
    try:
        out, _ = cap.generate_caption(Path("/x.jpg"), "Ningbo, China", "19:50", _cfg())
    finally:
        restore()
    assert out == "line one line two spaced"  # collapsed whitespace, no length cap


def test_emoji_field_appended():
    _captured, restore = _patch(({"caption": "Noodles for lunch", "emoji": "🍜"}, 0.1, True, True))
    try:
        out, _ = cap.generate_caption(Path("/x.jpg"), "Ningbo, China", "12:00", _cfg())
    finally:
        restore()
    assert out == "Noodles for lunch 🍜"


def test_emoji_field_junk_duplicate_and_cjk():
    for result, expected in [
        ({"caption": "Nice lane 🌿", "emoji": "🍜"}, "Nice lane 🌿"),      # already ends with one
        ({"caption": "Nice lane", "emoji": "abc"}, "Nice lane"),           # ascii junk skipped
        ({"caption": "Nice lane", "emoji": ""}, "Nice lane"),              # empty skipped
        ({"caption": "Nice lane"}, "Nice lane"),                           # field absent
        ({"caption": "Nice lane", "emoji": "🍜 for the noodles"}, "Nice lane 🍜"),  # padded field
        ({"caption": "Locals love 换购", "emoji": "🏮"}, "Locals love 换购 🏮"),  # CJK ≠ emoji
    ]:
        _captured, restore = _patch((result, 0.1, True, True))
        try:
            out, _ = cap.generate_caption(Path("/x.jpg"), None, None, _cfg())
        finally:
            restore()
        assert out == expected, (result, out)


def test_fails_safe_on_error():
    _captured, restore = _patch(({"caption": ""}, 0.1, False, False))  # ok=False
    try:
        out, _ = cap.generate_caption(Path("/x.jpg"), None, None, _cfg())
    finally:
        restore()
    assert out is None


def test_skips_on_ollama():
    orig = cap.provider_for_model
    cap.provider_for_model = lambda m: "ollama"
    try:
        out, used = cap.generate_caption(Path("/x.jpg"), "Ningbo", "10:00", _cfg())
    finally:
        cap.provider_for_model = orig
    assert out is None and used == 0


# ── Best-of-N (caption_candidates > 1) ──────────────────────────────────────────

def _cfg_n(n=4):
    c = _cfg()
    c.caption_candidates = n
    c.caption_picker_model = "qwen3.7-plus"
    return c


def _patch_multi(gen_results, pick_result):
    """Thread-safe fake: per-call caption results by label; captures the picker prompt."""
    import threading
    lock = threading.Lock()
    captured = {"gen_calls": 0, "pick_calls": 0, "pick_prompt": None}
    orig = (cap.call_vision_judge, cap.provider_for_model, cap.parse_model_effort)

    def fake_call(*, model_string, ollama_base_url, call):
        with lock:
            if call.label == "caption":
                r = gen_results[captured["gen_calls"] % len(gen_results)]
                captured["gen_calls"] += 1
                return r
            captured["pick_calls"] += 1
            captured["pick_prompt"] = call.prompt
            return pick_result(call.prompt) if callable(pick_result) else pick_result

    cap.call_vision_judge = fake_call
    cap.provider_for_model = lambda m: "dashscope"
    cap.parse_model_effort = lambda s: (s, None)

    def restore():
        cap.call_vision_judge, cap.provider_for_model, cap.parse_model_effort = orig

    return captured, restore


def test_best_of_n_picks_winner_and_counts_calls():
    gens = [({"caption": f"candidate {i}", "emoji": "🍜"}, 0.1, True, True) for i in range(4)]
    captured, restore = _patch_multi(gens, ({"best_index": 2, "reason": "best"}, 0.1, True, True))
    try:
        out, used = cap.generate_caption(Path("/x.jpg"), "Ningbo, China", "12:00", _cfg_n(4))
    finally:
        restore()
    # thread scheduling shuffles which text lands at which index — verify the
    # winner via the picker's own numbered list (index 2 → that exact text)
    line2 = next(ln for ln in captured["pick_prompt"].splitlines() if ln.startswith("2: "))
    assert out == line2[3:]
    assert used == 5  # 4 generators + 1 picker
    assert captured["gen_calls"] == 4 and captured["pick_calls"] == 1
    for i in range(4):  # all 4 distinct candidates reached the picker
        assert f"candidate {i} 🍜" in captured["pick_prompt"]


def test_best_of_n_identical_candidates_skip_picker():
    gens = [({"caption": "same text", "emoji": "🍜"}, 0.1, True, True)] * 4
    captured, restore = _patch_multi(gens, ({"best_index": 3}, 0.1, True, True))
    try:
        out, used = cap.generate_caption(Path("/x.jpg"), None, None, _cfg_n(4))
    finally:
        restore()
    assert out == "same text 🍜" and used == 4
    assert captured["pick_calls"] == 0  # deduped to one → no picker needed


def test_best_of_n_picker_failure_falls_back_to_first():
    gens = [({"caption": f"cand {i}"}, 0.1, True, True) for i in range(4)]
    for bad in [({"best_index": 99, "reason": "oob"}, 0.1, True, True),
                ({"nope": 1}, 0.1, True, True),
                ({"best_index": 0}, 0.1, False, False)]:  # ok=False
        _captured, restore = _patch_multi(gens, bad)
        try:
            out, _used = cap.generate_caption(Path("/x.jpg"), None, None, _cfg_n(4))
        finally:
            restore()
        assert out == "cand 0", (bad, out)


def test_winner_without_emoji_borrows_consensus_emoji():
    gens = [({"caption": "Lantern alley one", "emoji": "🏮"}, 0.1, True, True),
            ({"caption": "Lantern alley two", "emoji": "🏮"}, 0.1, True, True),
            ({"caption": "Lantern alley three", "emoji": "🍜"}, 0.1, True, True),
            ({"caption": "Accurate translation text", "emoji": ""}, 0.1, True, True)]

    def pick_no_emoji(prompt):  # picker chooses the emoji-less candidate, race-proof
        for ln in prompt.splitlines():
            head, _, text = ln.partition(": ")
            if head.isdigit() and text and not cap._is_emoji_char(text[-1]):
                return {"best_index": int(head), "reason": "most accurate"}, 0.1, True, True
        raise AssertionError("no emoji-less candidate in picker prompt")

    _captured, restore = _patch_multi(gens, pick_no_emoji)
    try:
        out, _used = cap.generate_caption(Path("/x.jpg"), None, None, _cfg_n(4))
    finally:
        restore()
    assert out == "Accurate translation text 🏮"  # consensus emoji borrowed, majority wins


def test_best_of_n_all_failed_returns_none():
    gens = [({"caption": ""}, 0.1, False, True)] * 4  # every generator errored
    captured, restore = _patch_multi(gens, ({"best_index": 0}, 0.1, True, True))
    try:
        out, used = cap.generate_caption(Path("/x.jpg"), None, None, _cfg_n(4))
    finally:
        restore()
    assert out is None and used == 4 and captured["pick_calls"] == 0


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {t.__name__}: {exc}"); traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
