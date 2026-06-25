"""
Cheap, local, adaptive image polish (CPU-only PIL + numpy; no GPU / AI / cost).

Measure-then-correct: each stage computes a cheap statistic on a 256px thumbnail
and scales its strength by how far the image is from "fine", so a well-exposed,
well-balanced, or deliberately-atmospheric photo (e.g. a hazy sunset) comes out
essentially unchanged. Fail-open: any error returns the input image untouched.

Pipeline order matters: white-balance → auto-levels → exposure → saturation →
sharpen. Per-stage strength comes from a per-intensity preset gain table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger("ic2x.polish")


@dataclass
class _Gains:
    white_balance: float
    auto_levels: float
    exposure: float
    saturation: float
    sharpen: float


# Per-stage strength ceilings. Each is multiplied by an adaptiveness ramp, so
# even "punchy" leaves an already-good photo nearly untouched.
_PRESETS: dict[str, _Gains] = {
    "natural": _Gains(white_balance=0.6, auto_levels=0.40, exposure=0.7, saturation=0.4, sharpen=0.0),
    "punchy":  _Gains(white_balance=1.0, auto_levels=0.65, exposure=1.0, saturation=1.0, sharpen=0.9),
}


@dataclass
class _Stats:
    cast_gains: tuple[float, float, float]  # per-channel gray-world gains
    cast_mag: float                          # max |gain - 1|
    luma_median: float
    luma_lo: float                           # 0.5th percentile
    luma_hi: float                           # 99.5th percentile
    sat_level: float                         # mean chroma / 255


def _measure(img: Image.Image) -> _Stats:
    """All statistics from a single 256px thumbnail (fixed cost vs source res)."""
    small = img.convert("RGB")
    small.thumbnail((256, 256), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    rm, gm, bm = float(r.mean()), float(g.mean()), float(b.mean())
    m = (rm + gm + bm) / 3.0
    gains = (m / max(rm, 1e-3), m / max(gm, 1e-3), m / max(bm, 1e-3))
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    return _Stats(
        cast_gains=gains,
        cast_mag=max(abs(x - 1.0) for x in gains),
        luma_median=float(np.median(luma)),
        luma_lo=float(np.percentile(luma, 0.5)),
        luma_hi=float(np.percentile(luma, 99.5)),
        sat_level=float(chroma.mean() / 255.0),
    )


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 below lo, 1 above hi, linear between."""
    if hi <= lo:
        return 1.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _white_balance(img: Image.Image, s: _Stats, k: float) -> Image.Image:
    # near-neutral already → no-op; very strong cast (deliberate wash) → leave it
    if s.cast_mag <= 0.06:
        return img
    strength = k * 0.5 * _ramp(s.cast_mag, 0.06, 0.35)
    applied = [1.0 + (gp - 1.0) * strength for gp in s.cast_gains]
    # warm-bias: damp corrections that REMOVE warmth (pulling red down / blue up)
    if applied[0] < 1.0:
        applied[0] = 1.0 + (applied[0] - 1.0) * 0.6
    if applied[2] > 1.0:
        applied[2] = 1.0 + (applied[2] - 1.0) * 0.6
    arr = np.asarray(img, dtype=np.float32) * np.array(applied, dtype=np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _auto_levels(img: Image.Image, s: _Stats, k: float) -> Image.Image:
    # already spans the full range → autocontrast would only risk clipping
    if s.luma_hi > 245 and s.luma_lo < 12:
        return img
    # preserve_tone keeps the stretch on luma (no colour shift); gentler on highlights
    ac = ImageOps.autocontrast(img, cutoff=(0.5, 0.2), preserve_tone=True)
    return Image.blend(img, ac, alpha=min(k, 0.7))


def _exposure(img: Image.Image, s: _Stats, k: float) -> Image.Image:
    med = s.luma_median
    if 105.0 <= med <= 140.0:  # well-exposed (and moody-dark sits just below) → leave
        return img
    factor = 122.0 / max(med, 1e-3)
    factor = 1.0 + (factor - 1.0) * (k * 0.4)
    factor = min(factor, 1.12) if factor > 1.0 else max(factor, 0.90)  # lift < darken
    return ImageEnhance.Brightness(img).enhance(factor)


def _saturation(img: Image.Image, s: _Stats, k: float) -> Image.Image:
    boost = k * (1.0 - _ramp(s.sat_level, 0.10, 0.22))  # max when grey, 0 when vivid
    factor = 1.0 + boost * 0.18
    if abs(factor - 1.0) < 1e-3:
        return img
    return ImageEnhance.Color(img).enhance(factor)


def _sharpen(img: Image.Image, k: float) -> Image.Image:
    if k <= 0:
        return img
    radius = 1.5 if max(img.size) >= 2000 else 1.0
    # threshold=3 leaves flat/noisy regions untouched (JPEG-artifact guard)
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(60 * k), threshold=3))


def polish(img: Image.Image, cfg) -> Image.Image:
    """Adaptive subtle polish. Returns a new RGB Image, or the input unchanged
    if disabled / intensity off / grayscale / any error. Never raises."""
    try:
        if not getattr(cfg, "polish_enabled", False):
            return img
        intensity = (getattr(cfg, "polish_intensity", "natural") or "natural").strip().lower()
        if intensity in ("off", "none", ""):
            return img
        gains = _PRESETS.get(intensity)
        if gains is None:
            logger.warning("polish: unknown intensity %r — skipping", intensity)
            return img
        if img.mode == "L":
            return img  # grayscale: colour ops don't apply
        work = img if img.mode == "RGB" else img.convert("RGB")
        s = _measure(work)
        if gains.white_balance > 0:
            work = _white_balance(work, s, gains.white_balance)
        if gains.auto_levels > 0:
            work = _auto_levels(work, s, gains.auto_levels)
        if gains.exposure > 0:
            work = _exposure(work, s, gains.exposure)
        if gains.saturation > 0:
            work = _saturation(work, s, gains.saturation)
        if gains.sharpen > 0:
            work = _sharpen(work, gains.sharpen)
        return work
    except Exception as exc:  # noqa: BLE001 — never block a post
        logger.warning("polish: skipped (%s)", exc)
        return img
