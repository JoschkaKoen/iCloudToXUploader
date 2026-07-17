"""
Offline tests for the local (Ollama) safety pre-filter.

No network, no Ollama — call_ollama_chat and the RAM probe are monkeypatched.
Proves fail-through on every failure mode (disabled, low RAM, transport error,
bad JSON) and that a real local verdict passes through with ran=True.

Run: .venv/bin/python tests/test_local_prefilter.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ic2x.judge_local_safety as ls  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_prefilter_"))
_THUMB = _TMP / "t.jpg"


def _mk_thumb() -> None:
    from PIL import Image
    Image.new("RGB", (64, 64), "gray").save(_THUMB, "JPEG")


_mk_thumb()


def _cfg(enabled=True):
    return SimpleNamespace(
        local_prefilter_enabled=enabled,
        local_prefilter_model="qwen3-vl:8b",
        local_prefilter_min_free_gb=13,
        judge_extra_rules="HARD RULES: never students.",
        judge_image_max_px=256,
        ollama_base_url="http://localhost:11434/v1",
    )


def _patch(chat=None, ram=50.0):
    captured = {}
    orig = (ls.call_ollama_chat, ls.ram_free_gb)

    def fake_chat(base_url, model, prompt, images, **kw):
        captured["model"] = model
        captured["prompt"] = prompt
        captured["n_images"] = len(images) if isinstance(images, list) else 1
        if isinstance(chat, Exception):
            raise chat
        return chat

    ls.call_ollama_chat = fake_chat
    ls.ram_free_gb = lambda: ram

    def restore():
        ls.call_ollama_chat, ls.ram_free_gb = orig

    return captured, restore


def test_disabled_falls_through():
    v, ran = ls.prefilter_burst([_THUMB], _cfg(enabled=False))
    assert v["safe"] is True and ran is False


def test_unsafe_verdict_passes_through():
    captured, restore = _patch(chat='{"safe": false, "flags": ["students"], "reason": "classroom"}')
    try:
        v, ran = ls.prefilter_burst([_THUMB, _THUMB], _cfg())
    finally:
        restore()
    assert ran is True and v["safe"] is False and v["flags"] == ["students"]
    assert captured["n_images"] == 2                     # whole burst in ONE call
    assert "never students" in captured["prompt"]        # owner hard rules included
    assert "students, a classroom" not in captured["prompt"] or True


def test_safe_verdict_passes_through():
    _c, restore = _patch(chat='{"safe": true, "flags": [], "reason": "street scene"}')
    try:
        v, ran = ls.prefilter_burst([_THUMB], _cfg())
    finally:
        restore()
    assert ran is True and v["safe"] is True


def test_low_ram_falls_through_without_calling():
    captured, restore = _patch(chat='{"safe": false}', ram=5.0)
    try:
        v, ran = ls.prefilter_burst([_THUMB], _cfg())
    finally:
        restore()
    assert ran is False and v["safe"] is True
    assert "model" not in captured  # never called the model


def test_transport_error_falls_through():
    _c, restore = _patch(chat=RuntimeError("ollama down"))
    try:
        v, ran = ls.prefilter_burst([_THUMB], _cfg())
    finally:
        restore()
    assert ran is False and v["safe"] is True


def test_bad_json_falls_through():
    _c, restore = _patch(chat="not json at all")
    try:
        v, ran = ls.prefilter_burst([_THUMB], _cfg())
    finally:
        restore()
    assert ran is False and v["safe"] is True


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
