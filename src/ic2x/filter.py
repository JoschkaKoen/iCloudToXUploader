"""
Screenshot detection filter.

Format filtering is intentionally omitted: icloudpd with --skip-videos only
delivers photos, and any PIL open failure (e.g. RAW files) is caught by the
per-file try/except in run.py.

Screenshot detection requires ALL THREE signals to fire before rejecting,
to avoid false positives on legitimate photos of screens.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger("ic2x.filter")

# Common iPhone / iPad screen resolutions (portrait and landscape).
# Both orientations are listed explicitly.
_SCREEN_RESOLUTIONS: set[tuple[int, int]] = {
    # iPhone 15 Pro / 15 Pro Max
    (1179, 2556), (2556, 1179),
    (1290, 2796), (2796, 1290),
    # iPhone 14 Pro / 14 Pro Max
    (1179, 2556), (2556, 1179),
    (1284, 2778), (2778, 1284),
    # iPhone 13 / 14
    (1170, 2532), (2532, 1170),
    # iPhone SE (3rd gen) / 12 mini / 13 mini
    (750, 1334), (1334, 750),
    (1080, 2340), (2340, 1080),
    # iPad Pro 12.9"
    (2048, 2732), (2732, 2048),
    # iPad Pro 11"
    (1668, 2388), (2388, 1668),
    # iPad Air
    (1640, 2360), (2360, 1640),
    # iPad mini
    (1488, 2266), (2266, 1488),
}


def is_screenshot(path: Path, album_name: str | None = None) -> tuple[bool, str]:
    """
    Return (True, reason) if the file looks like a device screenshot, else (False, "").

    All three signals must fire:
      1. No Make/Model EXIF tag (no camera metadata)
      2. Dimensions match a known device screen resolution
      3. Album name is "Screenshots" (when metadata is available)

    When album_name is None (icloudpd didn't provide it), signal 3 is skipped
    and we only require signals 1 and 2 — a conservative fallback.
    """
    try:
        with Image.open(path) as img:
            exif_data = img.getexif() or {}
            dims = img.size  # (width, height)

        # Signal 1: no camera make/model
        MAKE  = 271
        MODEL = 272
        has_camera = exif_data.get(MAKE) or exif_data.get(MODEL)
        if has_camera:
            return False, ""

        # Signal 2: dimensions match a known screen resolution
        if dims not in _SCREEN_RESOLUTIONS:
            return False, ""

        # Signal 3 (optional): album metadata
        if album_name is not None:
            if album_name.lower() != "screenshots":
                return False, ""
            return True, f"screenshot (album={album_name}, dims={dims})"

        # album_name unknown — signals 1+2 are enough
        return True, f"screenshot (no-camera EXIF, dims={dims})"

    except Exception as exc:
        logger.debug("filter: could not inspect %s: %s", path.name, exc)
        return False, ""
