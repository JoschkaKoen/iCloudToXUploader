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
    assert len(cap._CAPTION_PROMPT) < 1200         # one focused template, not a rule pile


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
    assert out == text and used is True
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
    assert out is None and used is False


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
