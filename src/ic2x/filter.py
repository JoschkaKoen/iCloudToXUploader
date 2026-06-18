"""
Screenshot / non-camera detection filter.

Secondary screenshot gate, run on the winner's full-res ORIGINAL (the primary
gate is Apple's Screenshots smart-album membership, applied to thumbnails during
burst assembly). The original retains the camera's Make/Model EXIF; screenshots
and images saved from other apps never have these tags, so their absence rejects
a file as non-camera.

Fail-closed: because this guards an autonomous public post, any inspection error
is treated as "reject" rather than waved through.
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
        logger.warning("filter: could not inspect %s: %s — rejecting (fail-closed)", path.name, exc)
        return True, f"exif inspection failed: {exc}"
