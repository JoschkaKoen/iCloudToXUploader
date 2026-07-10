"""
Offline tests for the pick-4 rotation judge. No network.

Proves: (1) pick-4 builds the 4 CW rotation candidates in the deterministic
shuffled order and maps the model's picked index back to the right CW degrees;
(2) a non-zero pick triggers the confirm stage with [as-shot, picked] and only
an apply_fix=true verdict rotates; (3) every failure path (not confident, bad
index, malformed/failed response, exception, confirm veto) keeps the photo
unchanged; (4) call_rotation delegates to pick-4 and Ollama models fall back to
the single-image method.

Run: .venv/bin/python tests/test_judge_rotation.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x import judge_rotation as jr  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_rot_test_"))


def _cfg() -> SimpleNamespace:
    work = _TMP / "work"
    work.mkdir(exist_ok=True)
    return SimpleNamespace(
        rotation_model="qwen3.5-flash, 800, 1200",
        rotation_image_max_px=1024,
        work_dir=work,
        ollama_base_url="http://localhost:11434/v1",
    )


def _base(name: str = "base.jpg") -> Path:
    """64x32 landscape: top half red, bottom half blue — orientation readable."""
    p = _TMP / name
    im = Image.new("RGB", (64, 32), (255, 0, 0))
    im.paste((0, 0, 255), (0, 16, 64, 32))
    im.save(p, "JPEG", quality=95)
    return p


def _dominant(px: tuple) -> str:
    return "red" if px[0] > px[2] else "blue"


def _top_color(path: Path) -> str:
    with Image.open(path) as im:
        w, h = im.size
        return _dominant(im.getpixel((w // 2, h // 8)))


class _Recorder:
    """Fake call_vision_judge_multi: scripted responses + captured calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, *, model_string, ollama_base_url, call, usage_out=None):
        # capture candidate pixels NOW — the tempdir dies when pick4 returns
        self.calls.append({
            "label": call.label,
            "prompt": call.prompt,
            "n": len(call.image_paths),
            "sizes": [Image.open(p).size for p in call.image_paths],
            "tops": [_top_color(p) for p in call.image_paths],
        })
        parsed, ok = self.responses.pop(0)
        return parsed, 0.01, ok, True


def _run_pick4(responses, name="base.jpg"):
    rec = _Recorder(responses)
    orig = jr.call_vision_judge_multi
    jr.call_vision_judge_multi = rec
    try:
        res, _el, _net = jr.call_rotation_pick4(_base(name), _cfg())
    finally:
        jr.call_vision_judge_multi = orig
    return res, rec


def test_candidates_follow_shuffle_order():
    order = jr._pick4_shuffle("base.jpg")
    # pick the 0° candidate → upright, no confirm stage
    res, rec = _run_pick4([({"upright_index": order.index(0), "confident": True}, True)])
    assert res == {"upright": True, "rotate_cw_degrees": 0}
    assert len(rec.calls) == 1 and rec.calls[0]["label"] == "rotation"
    assert rec.calls[0]["n"] == 4
    # geometry per candidate: 90/270 swap w/h; top half color identifies 0 vs 180
    for i, deg in enumerate(order):
        w, h = rec.calls[0]["sizes"][i]
        assert (w > h) == (deg in (0, 180)), (deg, w, h)
        if deg == 0:
            assert rec.calls[0]["tops"][i] == "red"
        if deg == 180:
            assert rec.calls[0]["tops"][i] == "blue"


def test_pick_plus_confirm_applies_fix():
    order = jr._pick4_shuffle("base.jpg")
    idx180 = order.index(180)
    res, rec = _run_pick4([
        ({"upright_index": idx180, "confident": True, "reason": "sky below"}, True),
        ({"apply_fix": True, "reason": "clearly upside down"}, True),
    ])
    assert res == {"upright": False, "rotate_cw_degrees": 180}
    assert [c["label"] for c in rec.calls] == ["rotation", "rotation_confirm"]
    confirm = rec.calls[1]
    assert confirm["n"] == 2
    assert confirm["tops"] == ["red", "blue"]        # [as-shot, picked 180°]
    assert "180 degrees clockwise" in confirm["prompt"]


def test_confirm_veto_keeps_photo():
    order = jr._pick4_shuffle("base.jpg")
    res, rec = _run_pick4([
        ({"upright_index": order.index(90), "confident": True}, True),
        ({"apply_fix": False, "reason": "top-down shot"}, True),
    ])
    assert res == {"upright": True, "rotate_cw_degrees": 0}
    assert len(rec.calls) == 2


def test_confirm_failure_keeps_photo():
    order = jr._pick4_shuffle("base.jpg")
    res, _rec = _run_pick4([
        ({"upright_index": order.index(270), "confident": True}, True),
        ({"apply_fix": False, "reason": "confirm failed"}, False),   # ok=False
    ])
    assert res == {"upright": True, "rotate_cw_degrees": 0}


def test_not_confident_keeps_photo():
    order = jr._pick4_shuffle("base.jpg")
    res, rec = _run_pick4([({"upright_index": order.index(180), "confident": False}, True)])
    assert res == {"upright": True, "rotate_cw_degrees": 0}
    assert len(rec.calls) == 1                        # no confirm stage


def test_bad_shapes_fail_open():
    for parsed, ok in (
        ({"nope": 1}, True),                          # missing upright_index
        ({"upright_index": "x", "confident": True}, True),   # non-int
        ({"upright_index": 7, "confident": True}, True),     # out of range
        ({"upright": True, "rotate_cw_degrees": 0}, False),  # ok=False (fail_value)
    ):
        res, _rec = _run_pick4([(parsed, ok)])
        assert res == {"upright": True, "rotate_cw_degrees": 0}, parsed


def test_exception_fails_open():
    def boom(**_k):
        raise RuntimeError("network down")
    orig = jr.call_vision_judge_multi
    jr.call_vision_judge_multi = boom
    try:
        res, _el, _net = jr.call_rotation_pick4(_base(), _cfg())
    finally:
        jr.call_vision_judge_multi = orig
    assert res == {"upright": True, "rotate_cw_degrees": 0}


def test_shuffle_deterministic_full_permutation():
    for name in ("a.jpg", "b.jpg", "IMG_1234.jpg"):
        order = jr._pick4_shuffle(name)
        assert order == jr._pick4_shuffle(name)
        assert sorted(order) == [0, 90, 180, 270]
    assert any(jr._pick4_shuffle(n) != jr._pick4_shuffle("a.jpg")
               for n in ("b.jpg", "c.jpg", "d.jpg"))


def test_call_rotation_delegates_to_pick4():
    sentinel = ({"upright": False, "rotate_cw_degrees": 90}, 0.5, True)
    orig = jr.call_rotation_pick4
    jr.call_rotation_pick4 = lambda p, c, model_string=None: sentinel
    try:
        assert jr.call_rotation(Path("x.jpg"), _cfg()) == sentinel
    finally:
        jr.call_rotation_pick4 = orig


def test_ollama_model_falls_back_to_single():
    seen = {}

    def fake_single(path, cfg, model_string=None):
        seen["called"] = True
        return {"upright": True, "rotate_cw_degrees": 0}, 0.0, False

    orig_p, orig_s = jr.provider_for_model, jr.call_rotation_single
    jr.provider_for_model = lambda m: "ollama"
    jr.call_rotation_single = fake_single
    try:
        res, _el, _net = jr.call_rotation_pick4(_base(), _cfg())
    finally:
        jr.provider_for_model, jr.call_rotation_single = orig_p, orig_s
    assert seen.get("called") and res["upright"]


def _main() -> int:
    failed = 0
    for name, t in sorted(globals().items()):
        if name.startswith("test_") and callable(t):
            try:
                t(); print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                import traceback; print(f"FAIL {name}: {exc}"); traceback.print_exc()
    print("OK" if not failed else "FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
