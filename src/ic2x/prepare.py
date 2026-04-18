"""
Prepare an image for the queue:
  1. exif_transpose — bake rotation into pixels
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


def prepare(src_path: Path, dest_dir: Path, phash: str) -> Path:
    """
    Rotate, re-encode to JPEG, strip EXIF.
    Returns the path of the prepared file in dest_dir.
    """
    out = dest_dir / f"{phash}.jpg"
    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(out, "JPEG", quality=92, exif=b"")
    logger.debug("prepare: %s → %s", src_path.name, out.name)
    return out
