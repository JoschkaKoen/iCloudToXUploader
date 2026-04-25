"""
Screenshot detection filter.

Format filtering is intentionally omitted: icloudpd with --skip-videos only
delivers photos, and any PIL open failure (e.g. RAW files) is caught by the
per-file try/except in run.py.

Every real photo pulled from iCloud carries Make/Model EXIF written by the
camera.  Screenshots and images saved from other apps never have these tags.
Absence of Make/Model EXIF is therefore sufficient to reject an image as a
non-camera file — no resolution allowlist needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger("ic2x.filter")


_EXIF_TAG_MAKE  = 271
_EXIF_TAG_MODEL = 272


def is_screenshot(path: Path) -> tuple[bool, str]:
    """Return (True, reason) if the file has no camera EXIF, else (False, "").

    iCloud preserves the original Make/Model EXIF for every camera photo.
    Any file that lacks these tags was not shot by a camera (screenshot,
    saved image, etc.) and is rejected.
    """
    try:
        with Image.open(path) as img:
            exif_data = img.getexif() or {}
        if exif_data.get(_EXIF_TAG_MAKE) or exif_data.get(_EXIF_TAG_MODEL):
            return False, ""
        return True, "no camera EXIF (Make/Model absent)"
    except Exception as exc:
        logger.debug("filter: could not inspect %s: %s", path.name, exc)
        return False, ""
