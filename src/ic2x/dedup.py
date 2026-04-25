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
from PIL import Image

from ic2x.utils.image_utils import oriented

logger = logging.getLogger("ic2x.dedup")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phash_of(path: Path) -> str:
    with Image.open(path) as img:
        return str(imagehash.phash(oriented(img)))
