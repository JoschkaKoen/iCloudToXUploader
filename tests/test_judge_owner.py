"""
Offline tests for the owner-selfie gate (judge_owner).

No network — call_vision_judge_multi is monkeypatched. Proves: refs + candidate
are sent in the right order (candidate LAST, refs capped at 3); a hit requires
owner_present AND main_subject; every failure mode (no refs, bad shape, call
error, ollama model) fails OPEN to "not a selfie" so posting is never blocked.

Run: .venv/bin/python tests/test_judge_owner.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ic2x.judge_owner as jo  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_owner_test_"))
_CAND = _TMP / "candidate.jpg"
_CAND.write_bytes(b"fake-jpeg")


def _mk_refs(n: int) -> Path:
    d = _TMP / f"refs{n}_{os.urandom(2).hex()}"
    d.mkdir()
    for i in range(n):
        p = d / f"owner{i}.jpg"
        p.write_bytes(b"fake-jpeg")
        os.utime(p, (1000 + i, 1000 + i))  # deterministic mtime order
    return d


def _cfg(refs_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        owner_check_enabled=True,
        owner_check_model="qwen3.5-flash, 800, 1200",
        owner_check_image_max_px=256,
        owner_refs_dir=refs_dir,
        ollama_base_url="http://localhost:11434/v1",
    )


def _patch(result, ok=True):
    captured = {}
    orig = jo.call_vision_judge_multi

    def fake(*, model_string, ollama_base_url, call):
        captured["paths"] = list(call.image_paths)
        captured["prompt"] = call.prompt
        captured["label"] = call.label
        return result, 0.1, ok, True

    jo.call_vision_judge_multi = fake

    def restore():
        jo.call_vision_judge_multi = orig

    return captured, restore


def test_no_refs_skips_without_network():
    captured, restore = _patch({"owner_present": True, "main_subject": True})
    try:
        res, _el, used = jo.call_owner_check(_CAND, _cfg(_TMP / "missing"))
    finally:
        restore()
    assert res == {"owner_main_subject": False, "reason": ""}
    assert used is False and "paths" not in captured  # never called


def test_hit_rejects_and_sends_candidate_last():
    refs = _mk_refs(2)
    captured, restore = _patch(
        {"owner_present": True, "main_subject": True, "reason": "same face, selfie"})
    try:
        res, _el, used = jo.call_owner_check(_CAND, _cfg(refs))
    finally:
        restore()
    assert res["owner_main_subject"] is True and "same face" in res["reason"]
    assert used is True
    assert captured["paths"][-1] == _CAND and len(captured["paths"]) == 3
    assert all(p.parent == refs for p in captured["paths"][:-1])
    assert "first 2 image(s)" in captured["prompt"]
    assert captured["label"] == "owner_check"


def test_owner_in_background_passes():
    refs = _mk_refs(1)
    _captured, restore = _patch(
        {"owner_present": True, "main_subject": False, "reason": "tiny in background"})
    try:
        res, _el, _u = jo.call_owner_check(_CAND, _cfg(refs))
    finally:
        restore()
    assert res["owner_main_subject"] is False


def test_stranger_close_up_passes():
    refs = _mk_refs(1)
    _captured, restore = _patch(
        {"owner_present": False, "main_subject": True, "reason": "different person"})
    try:
        res, _el, _u = jo.call_owner_check(_CAND, _cfg(refs))
    finally:
        restore()
    assert res["owner_main_subject"] is False


def test_bad_shape_and_call_error_fail_open():
    refs = _mk_refs(1)
    for result, ok in [({"nonsense": 1}, True),
                       ({"owner_main_subject": False, "reason": ""}, False)]:
        _captured, restore = _patch(result, ok=ok)
        try:
            res, _el, _u = jo.call_owner_check(_CAND, _cfg(refs))
        finally:
            restore()
        assert res["owner_main_subject"] is False, result


def test_refs_capped_at_three_newest():
    refs = _mk_refs(5)
    captured, restore = _patch(
        {"owner_present": False, "main_subject": False, "reason": ""})
    try:
        jo.call_owner_check(_CAND, _cfg(refs))
    finally:
        restore()
    sent = captured["paths"]
    assert len(sent) == 4  # 3 refs + candidate
    # newest by mtime = owner4, owner3, owner2
    assert [p.name for p in sent[:-1]] == ["owner4.jpg", "owner3.jpg", "owner2.jpg"]
    assert "first 3 image(s)" in captured["prompt"]


def test_ollama_model_fails_open():
    refs = _mk_refs(1)
    orig = jo.provider_for_model
    jo.provider_for_model = lambda m: "ollama"
    captured, restore = _patch({"owner_present": True, "main_subject": True})
    try:
        res, _el, used = jo.call_owner_check(_CAND, _cfg(refs))
    finally:
        restore()
        jo.provider_for_model = orig
    assert res["owner_main_subject"] is False and used is False
    assert "paths" not in captured


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
