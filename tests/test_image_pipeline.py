"""
Offline tests for the local image pipeline — the safety-critical screenshot gate
(filter.is_screenshot, fail-closed), perceptual + exact dedup (dedup), the queue
re-encode (prepare, EXIF-stripping), and capture-time extraction (geo).

No iCloud / network / AI. Run: .venv/bin/python tests/test_image_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x import dedup, geo, prepare  # noqa: E402
from ic2x.filter import is_screenshot  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_imgpipe_"))


def _camera_jpeg(path: Path, color="red", *, make="Apple", model="iPhone 15", dto=None) -> Path:
    im = Image.new("RGB", (96, 64), color)
    ex = Image.Exif()
    if make:
        ex[271] = make
    if model:
        ex[272] = model
    if dto:
        sub = ex.get_ifd(0x8769)
        sub[36867] = dto
        ex[0x8769] = sub
    im.save(path, "JPEG", exif=ex.tobytes())
    return path


def _bare_jpeg(path: Path, color="blue") -> Path:
    Image.new("RGB", (96, 64), color).save(path, "JPEG")  # no EXIF
    return path


# ── filter.is_screenshot (fail-closed) ───────────────────────────────────────

def test_camera_exif_is_not_screenshot():
    p = _camera_jpeg(_TMP / "cam.jpg")
    is_ss, reason = is_screenshot(p)
    assert is_ss is False and reason == ""


def test_no_camera_exif_is_screenshot():
    p = _bare_jpeg(_TMP / "bare.jpg")
    is_ss, reason = is_screenshot(p)
    assert is_ss is True and "Make/Model" in reason


def test_model_only_counts_as_camera():
    p = _camera_jpeg(_TMP / "modelonly.jpg", make=None, model="Pixel 9")
    assert is_screenshot(p)[0] is False


def test_unreadable_file_fails_closed():
    p = _TMP / "broken.jpg"
    p.write_bytes(b"not a real jpeg")
    is_ss, reason = is_screenshot(p)
    assert is_ss is True and reason  # fail-closed → rejected, with a reason


# ── dedup ────────────────────────────────────────────────────────────────────

def test_sha256_distinguishes_bytes():
    a = _camera_jpeg(_TMP / "a.jpg", "red")
    b = _camera_jpeg(_TMP / "b.jpg", "green")
    assert dedup.sha256_of(a) != dedup.sha256_of(b)
    assert dedup.sha256_of(a) == dedup.sha256_of(a)  # stable


def _split(path: Path, *, vertical: bool) -> Path:
    """A high-contrast edge image: left/right (vertical) or top/bottom (horizontal)
    white/black — two orientations with clearly different perceptual hashes."""
    im = Image.new("RGB", (64, 64), "black")
    px = im.load()
    for x in range(64):
        for y in range(64):
            if (x < 32) if vertical else (y < 32):
                px[x, y] = (255, 255, 255)
    im.save(path, "JPEG", quality=95)
    return path


def test_phash_stable_and_distinguishes_structure():
    v1 = _split(_TMP / "v1.jpg", vertical=True)
    v2 = _split(_TMP / "v2.jpg", vertical=True)
    h = _split(_TMP / "h.jpg", vertical=False)
    import imagehash
    hv1 = imagehash.hex_to_hash(dedup.phash_of(v1))
    hv2 = imagehash.hex_to_hash(dedup.phash_of(v2))
    hh = imagehash.hex_to_hash(dedup.phash_of(h))
    assert dedup.phash_of(v1) == dedup.phash_of(v1)        # deterministic
    assert (hv1 - hv2) <= 2, "identical scenes should hash near-identically"
    assert (hv1 - hh) > 2, "a structurally different scene should hash differently"


# ── prepare (queue re-encode, EXIF-stripped) ─────────────────────────────────

def test_prepare_outputs_phash_jpeg_without_exif():
    src = _camera_jpeg(_TMP / "src.jpg", "purple", dto="2026:06:27 19:50:31")
    ph = dedup.phash_of(src)
    out = prepare.prepare(src, _TMP / "queue", ph)
    assert out.exists() and out.name == f"{ph}.jpg"
    with Image.open(out) as im:
        assert im.format == "JPEG"
        ex = im.getexif()
        assert not ex.get(271) and not ex.get(272), "EXIF Make/Model must be stripped"
    # the prepared file is itself a 'screenshot' by the EXIF gate (no camera tags) —
    # which is why the gate runs on the ORIGINAL, before prepare strips EXIF.
    assert is_screenshot(out)[0] is True


# ── geo.capture_time ─────────────────────────────────────────────────────────

def test_capture_time_reads_local_hh_mm():
    p = _camera_jpeg(_TMP / "timed.jpg", dto="2026:06:27 09:05:00")
    assert geo.capture_time(p) == "09:05"


def test_capture_time_none_without_exif():
    assert geo.capture_time(_bare_jpeg(_TMP / "notime.jpg")) is None


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
