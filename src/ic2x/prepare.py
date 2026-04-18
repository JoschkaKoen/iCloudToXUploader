"""
Prepare an image for the queue:
  1. exif_transpose — bake rotation into pixels (JPEG EXIF + HEIC fallback)
  2. Re-encode as JPEG at quality=92
  3. Strip all EXIF metadata via exif=b""

Always opens the original source file fresh to avoid double-transpose.
Output filename is {phash}.jpg.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageOps

logger = logging.getLogger("ic2x.prepare")

_HEIF_OPS = {
    2: Image.FLIP_LEFT_RIGHT,
    3: Image.ROTATE_180,
    4: Image.FLIP_TOP_BOTTOM,
    5: Image.TRANSPOSE,
    6: Image.ROTATE_270,
    7: Image.TRANSVERSE,
    8: Image.ROTATE_90,
}


def _oriented(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation (JPEG) or HEIF orientation (HEIC) to img."""
    img = ImageOps.exif_transpose(img)          # standard JPEG EXIF path
    orientation = img.info.get("orientation")   # pillow_heif fallback (int 1-8)
    if orientation and orientation != 1:
        op = _HEIF_OPS.get(orientation)
        if op:
            img = img.transpose(op)
    return img


def prepare(src_path: Path, dest_dir: Path, phash: str) -> Path:
    """
    Rotate, re-encode to JPEG, strip EXIF.
    Returns the path of the prepared file in dest_dir.
    """
    out = dest_dir / f"{phash}.jpg"
    with Image.open(src_path) as img:
        img = _oriented(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(out, "JPEG", quality=92, exif=b"")
    logger.debug("prepare: %s → %s", src_path.name, out.name)
    return out
