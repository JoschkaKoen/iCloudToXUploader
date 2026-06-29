"""
Offline tests for the free local adaptive polish (ic2x/polish.py). No network, no
AI, no iCloud. Proves the opt-in gating, fail-open safety, adaptiveness (a
well-exposed neutral image is left ~unchanged), and that a clearly dark/flat image
is measurably lifted.

Run: .venv/bin/python tests/test_polish.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x import polish as P  # noqa: E402


def _cfg(enabled=True, intensity="natural"):
    return SimpleNamespace(polish_enabled=enabled, polish_intensity=intensity)


def _mean_luma(img: Image.Image) -> float:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    return float((0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean())


def _photo(seed: int = 0, scale: float = 1.0, base: int = 0) -> Image.Image:
    """A non-degenerate synthetic RGB 'photo' (gradient + texture), optionally
    darkened by `scale` and offset by `base`."""
    rng = np.random.default_rng(seed)
    h = w = 128
    grad = np.linspace(0, 255, w, dtype=np.float32)[None, :].repeat(h, 0)
    arr = np.stack([grad, grad * 0.8 + 20, grad * 0.6 + 40], axis=-1)
    arr += rng.normal(0, 12, arr.shape).astype(np.float32)
    arr = arr * scale + base
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


# ── gating / fail-open ───────────────────────────────────────────────────────

def test_disabled_returns_same_object():
    img = _photo()
    assert P.polish(img, _cfg(enabled=False)) is img


def test_intensity_off_returns_same_object():
    img = _photo()
    for off in ("off", "none", "  OFF  ", "None"):
        assert P.polish(img, _cfg(intensity=off)) is img


def test_empty_intensity_defaults_to_natural():
    # empty/unset intensity falls back to the "natural" preset (config default),
    # so it DOES polish rather than passing through.
    img = _photo()
    assert P.polish(img, _cfg(intensity="")) is not img


def test_unknown_intensity_returns_same_object():
    img = _photo()
    assert P.polish(img, _cfg(intensity="ludicrous")) is img


def test_grayscale_passthrough():
    img = _photo().convert("L")
    assert P.polish(img, _cfg()) is img


def test_never_raises_on_odd_inputs():
    for mode, size in (("P", (4, 4)), ("RGBA", (8, 8)), ("RGB", (1, 1))):
        img = Image.new(mode, size)
        out = P.polish(img, _cfg(intensity="punchy"))
        assert isinstance(out, Image.Image)


# ── adaptiveness / effect ────────────────────────────────────────────────────

def test_wellexposed_image_barely_changes():
    img = _photo()                       # already full-range, neutral-ish
    out = P.polish(img, _cfg(intensity="natural"))
    delta = abs(_mean_luma(out) - _mean_luma(img))
    assert delta < 18.0, f"natural polish moved a good image too much: {delta:.1f}"


def test_dark_flat_image_is_lifted():
    dark = _photo(scale=0.32, base=8)    # compressed into the shadows
    out = P.polish(dark, _cfg(intensity="punchy"))
    assert out is not dark
    assert _mean_luma(out) > _mean_luma(dark) + 2.0, "punchy polish should brighten a dark image"


def test_output_is_rgb_and_same_size():
    img = _photo()
    out = P.polish(img, _cfg(intensity="punchy"))
    assert out.mode == "RGB" and out.size == img.size


def test_bot_apply_polish_helper_brightens_file_in_place():
    """The bot's _apply_polish bakes polish into the prepared JPEG in place."""
    import tempfile

    from ic2x.bot import _apply_polish
    td = Path(tempfile.mkdtemp(prefix="ic2x_polish_"))
    p = td / "dark.jpg"
    _photo(scale=0.3, base=6).save(p, "JPEG", quality=92)
    before = _mean_luma(Image.open(p))
    _apply_polish(p, _cfg(intensity="punchy"))
    after = _mean_luma(Image.open(p))
    assert after > before + 2.0, f"_apply_polish should brighten the file ({before:.1f}→{after:.1f})"
    # still a valid, openable JPEG
    with Image.open(p) as im:
        im.verify()


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
