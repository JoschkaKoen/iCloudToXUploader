"""
GPS EXIF → "📍 City, Country" caption line.

The winner's iCloud *original* still carries GPS EXIF (thumbs/renditions do not),
so we read it here — before prepare()/rotation strip EXIF from the posted JPEG.
The post therefore shows the city as TEXT but never embeds exact coordinates.

Reverse-geocoding uses BigDataCloud's free, no-key reverse-geocode-client endpoint,
which returns a recognizable city (e.g. "Ningbo", not the "Jiangbei District" that
raw OSM/Nominatim gives). Wholly best-effort: no GPS, no network, or any error →
returns None and the caption posts without a location line. Cached by ~100 m coord.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path

from ic2x.utils.image_utils import register_heif_opener

register_heif_opener()
from PIL import ExifTags, Image  # noqa: E402 — must follow HEIF-opener registration

logger = logging.getLogger("ic2x.geo")

_GEOCODE_URL = "https://api.bigdatacloud.net/data/reverse-geocode-client"
_cache: dict[tuple[float, float], str | None] = {}

# Every other network call in a cycle retries (rotation 4×, caption 4×, create_tweet
# 6×) — this one did not, so a single blip cost the post its location permanently.
# Seen 2026-08-05 04:40: GPS read fine at 31.5778,120.2983 (Wuxi), the lookup failed
# once during a flaky window, and the post went out with no 📍 line at all.
#
# The retry then has to outlast the failure. A fixed 2s gap did not: on 2026-08-10 all
# three attempts fell inside FOUR seconds (08:19:41/43/45) and every one hit the same
# `SSL: UNEXPECTED_EOF_WHILE_READING`, so the post still lost its location. That error
# is GFW interference against a foreign API from the China host — measured 5/5 reachable
# moments later, so it is intermittent, not blocked, and it outlasts a 4s window.
# (nominatim.openstreetmap.org was checked as a fallback provider and is 0/5 from this
# host, so redundancy is not available — riding the blip out is.)
# Backoff 3s, 9s, 27s ≈ 40s total, which the 6.39h posting cadence absorbs easily.
_GEOCODE_ATTEMPTS = 4
_GEOCODE_RETRY_DELAY = 3.0
_GEOCODE_RETRY_FACTOR = 3.0


def _to_decimal(dms, ref) -> float:
    d, m, s = (float(x) for x in dms)
    val = d + m / 60.0 + s / 3600.0
    return -val if str(ref).upper() in ("S", "W") else val


def extract_gps(image_path: Path) -> tuple[float, float] | None:
    """(lat, lon) decimal degrees from the image's GPS EXIF, or None."""
    try:
        exif = Image.open(image_path).getexif()
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        if not gps:
            return None
        g = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
        if not g.get("GPSLatitude") or not g.get("GPSLongitude"):
            return None
        lat = _to_decimal(g["GPSLatitude"], g.get("GPSLatitudeRef", "N"))
        lon = _to_decimal(g["GPSLongitude"], g.get("GPSLongitudeRef", "E"))
        if abs(lat) < 1e-6 and abs(lon) < 1e-6:
            return None  # 0,0 is a non-fix, not a real location
        return lat, lon
    except Exception as exc:  # noqa: BLE001 — best-effort, never raise
        logger.debug("gps: extract failed for %s: %s",
                     getattr(image_path, "name", image_path), exc)
        return None


def reverse_geocode(lat: float, lon: float, timeout: float = 10.0) -> str | None:
    """"City, Country" for a coordinate (BigDataCloud), or None. Cached by ~100 m.

    Only SUCCESSFUL lookups are cached (including a genuine "no city here" → None).
    A transient failure (network/timeout) returns None WITHOUT caching, so the next
    post from that area retries instead of being permanently stuck for the life of
    the long-running bot process.

    Retried in close succession before giving up: losing the lookup costs the post
    BOTH its 📍 line and the caption model's location grounding, and the coordinate
    cache rarely rescues it — keyed at ~100 m, consecutive posts from the same city
    are usually different cells."""
    key = (round(lat, 3), round(lon, 3))
    if key in _cache:
        return _cache[key]
    url = f"{_GEOCODE_URL}?latitude={lat}&longitude={lon}&localityLanguage=en"
    for attempt in range(1, _GEOCODE_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                d = json.load(resp)
            city = d.get("city") or d.get("locality") or ""
            country = d.get("countryName") or ""
            place = ", ".join(p for p in (city, country) if p) or None
        except Exception as exc:  # noqa: BLE001 — best-effort, never raise (and don't cache)
            if attempt < _GEOCODE_ATTEMPTS:
                delay = _GEOCODE_RETRY_DELAY * (_GEOCODE_RETRY_FACTOR ** (attempt - 1))
                logger.warning("geo: reverse geocode failed for %.4f,%.4f "
                               "(attempt %d/%d), retrying in %.0fs: %s",
                               lat, lon, attempt, _GEOCODE_ATTEMPTS, delay, exc)
                time.sleep(delay)
                continue
            logger.warning("geo: reverse geocode failed for %.4f,%.4f after %d "
                           "attempts — posting without a location: %s",
                           lat, lon, _GEOCODE_ATTEMPTS, exc)
            return None
        _cache[key] = place
        return place
    return None


def capture_time(image_path: Path) -> str | None:
    """Local capture time as "HH:MM" from EXIF DateTimeOriginal, or None.

    EXIF stores capture time in the camera's LOCAL timezone, so the value is
    already correct for display next to the location — no conversion needed.
    """
    try:
        exif = Image.open(image_path).getexif()
        sub = exif.get_ifd(ExifTags.IFD.Exif)
        # 36867 DateTimeOriginal, 36868 DateTimeDigitized, 306 DateTime (fallbacks)
        raw = sub.get(36867) or sub.get(36868) or exif.get(306)
        if not raw:
            return None
        parts = str(raw).strip().split(" ")  # "YYYY:MM:DD HH:MM:SS"
        if len(parts) < 2:
            return None
        hh, mm = (parts[1].split(":") + ["", ""])[:2]
        if not (hh.isdigit() and mm.isdigit()):
            return None
        return f"{int(hh):02d}:{int(mm):02d}"
    except Exception as exc:  # noqa: BLE001 — best-effort, never raise
        logger.debug("time: extract failed for %s: %s",
                     getattr(image_path, "name", image_path), exc)
        return None


def place_and_time(image_path: Path, cfg) -> tuple[str | None, str | None]:
    """(place, time) for the image — e.g. ("Ningbo, China", "19:50"); either may be
    None. `place` needs GPS EXIF + a successful geocode; `time` needs EXIF capture
    time. Both are fed to the caption model so it can ground the caption in WHERE and
    WHEN, and `place`/`time` also build the 📍 line. Never raises."""
    if not getattr(cfg, "location_enabled", True):
        return None, None
    place: str | None = None
    try:
        gps = extract_gps(image_path)
        if gps:
            place = reverse_geocode(gps[0], gps[1],
                                    timeout=getattr(cfg, "location_timeout", 10.0))
    except Exception as exc:  # noqa: BLE001 — never block a post
        logger.warning("geo: place lookup failed: %s", exc)
    return place, capture_time(image_path)


def format_location_line(place: str | None, when: str | None) -> str | None:
    """"📍 HH:MM City, Country" (time before the city), or "📍 City, Country" if the
    time is unreadable, or None when there is no geocoded place."""
    if not place:
        return None
    return f"📍 {when} {place}" if when else f"📍 {place}"


def location_line(image_path: Path, cfg) -> str | None:
    """Back-compat convenience: the combined "📍 HH:MM City, Country" line, or None."""
    place, when = place_and_time(image_path, cfg)
    return format_location_line(place, when)
