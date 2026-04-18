"""
Duplicate detection: SHA-256 (exact) and pHash (perceptual).

SHA-256 is checked first — it requires no image decode.
pHash is computed after exif_transpose so the hash reflects the final
rotated pixel content, matching what prepare.py will save to queue/.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

logger = logging.getLogger("ic2x.dedup")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phash_of(path: Path) -> str:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return str(imagehash.phash(img))
