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
import urllib.request
from pathlib import Path

from ic2x.utils.image_utils import register_heif_opener

register_heif_opener()
from PIL import ExifTags, Image  # noqa: E402 — must follow HEIF-opener registration

logger = logging.getLogger("ic2x.geo")

_GEOCODE_URL = "https://api.bigdatacloud.net/data/reverse-geocode-client"
_cache: dict[tuple[float, float], str | None] = {}


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
    """"City, Country" for a coordinate (BigDataCloud), or None. Cached by ~100 m."""
    key = (round(lat, 3), round(lon, 3))
    if key in _cache:
        return _cache[key]
    url = f"{_GEOCODE_URL}?latitude={lat}&longitude={lon}&localityLanguage=en"
    place: str | None = None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            d = json.load(resp)
        city = d.get("city") or d.get("locality") or ""
        country = d.get("countryName") or ""
        place = ", ".join(p for p in (city, country) if p) or None
    except Exception as exc:  # noqa: BLE001 — best-effort, never raise
        logger.warning("geo: reverse geocode failed for %.4f,%.4f: %s", lat, lon, exc)
    _cache[key] = place
    return place


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


def location_line(image_path: Path, cfg) -> str | None:
    """"📍 HH:MM City, Country" for the image (time directly before the city),
    or None. Time is omitted if unreadable; the line needs a geocoded place.
    Never raises."""
    if not getattr(cfg, "location_enabled", True):
        return None
    try:
        gps = extract_gps(image_path)
        if not gps:
            return None
        place = reverse_geocode(gps[0], gps[1],
                                timeout=getattr(cfg, "location_timeout", 10.0))
        if not place:
            return None
        when = capture_time(image_path)
        return f"📍 {when} {place}" if when else f"📍 {place}"
    except Exception as exc:  # noqa: BLE001 — never block a post
        logger.warning("geo: location_line failed: %s", exc)
        return None
