"""
Offline tests for the local face gate (decision logic; no models needed).

The ONNX detector/recognizer are stubbed — these tests pin the DECISION rules:
owner match + prominent face → hit; owner only small in background → pass;
strangers → pass; no faces → pass; unavailable gate → None (VLM fallback).
An optional integration test runs only when the model files + owner refs exist.

Run: .venv/bin/python tests/test_face_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from ic2x.face_gate import MAIN_FACE_AREA, SIM_THRESHOLD, FaceGate  # noqa: E402


def _gate(faces, sims):
    """A FaceGate with stubbed internals: `faces` = list of (x, y, w, h) on a
    1000x1000 image; `sims` = owner similarity per face."""
    g = FaceGate.__new__(FaceGate)
    g.available = True
    g._read = lambda p: np.zeros((1000, 1000, 3), dtype=np.uint8)
    g._faces = lambda img: [np.array([x, y, w, h] + [0] * 11, dtype=float)
                            for x, y, w, h in faces]
    embs = iter(range(len(faces)))
    g._embed = lambda img, f: next(embs)
    g._owner_sim = lambda emb: sims[emb]
    return g


def test_owner_prominent_face_hits():
    g = _gate([(100, 100, 300, 300)], [0.71])           # 9% area, strong match
    hit, reason = g.owner_main_subject(Path("/x.jpg"))
    assert hit is True and "owner face" in reason


def test_owner_tiny_in_background_passes():
    g = _gate([(10, 10, 40, 40)], [0.80])               # 0.16% area — background
    hit, reason = g.owner_main_subject(Path("/x.jpg"))
    assert hit is False and "background" in reason


def test_strangers_pass_even_when_large():
    g = _gate([(0, 0, 500, 500), (600, 0, 350, 350)], [0.10, 0.05])
    hit, reason = g.owner_main_subject(Path("/x.jpg"))
    assert hit is False and "no owner match" in reason


def test_no_faces_passes():
    g = _gate([], [])
    hit, reason = g.owner_main_subject(Path("/x.jpg"))
    assert hit is False and "no faces" in reason


def test_unavailable_gate_returns_none():
    g = FaceGate.__new__(FaceGate)
    g.available = False
    assert g.owner_main_subject(Path("/x.jpg")) is None


def test_threshold_boundaries():
    area_side = int((MAIN_FACE_AREA * 1000 * 1000) ** 0.5) + 5   # just over area floor
    g = _gate([(0, 0, area_side, area_side)], [SIM_THRESHOLD + 0.01])
    hit, _ = g.owner_main_subject(Path("/x.jpg"))
    assert hit is True
    g = _gate([(0, 0, area_side, area_side)], [SIM_THRESHOLD - 0.01])
    hit, _ = g.owner_main_subject(Path("/x.jpg"))
    assert hit is False


def test_integration_if_models_present():
    root = Path(__file__).resolve().parent.parent
    models, refs = root / "models" / "face", root / "owner_refs"
    if not (models / "face_detection_yunet_2023mar.onnx").is_file() or not refs.is_dir():
        print("  (models/refs absent — integration part skipped)")
        return
    g = FaceGate(models, refs)
    if not g.available:
        print("  (gate unavailable — skipped)")
        return
    ref = next(iter(refs.glob("*.jpg")), None)
    if ref is not None:
        res = g.owner_main_subject(ref)
        assert res is not None and res[0] is True  # the reference IS the owner


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
