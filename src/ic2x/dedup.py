"""
Duplicate detection: SHA-256 (exact) and pHash (perceptual).

SHA-256 is checked first — it requires no image decode.
pHash is computed after orientation correction so the hash reflects the final
rotated pixel content, matching what prepare.py will save to queue/.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

logger = logging.getLogger("ic2x.dedup")

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


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phash_of(path: Path) -> str:
    with Image.open(path) as img:
        img = _oriented(img)
        return str(imagehash.phash(img))
