"""
iCloud photo access via the pyicloud library (replaces the old icloudpd CLI).

Exposes exactly what the bot needs: a non-interactive session (raising a typed
ReauthRequired when 2FA is due, so the loop notifies instead of blocking), a
metadata-only newest-first iterator over still images, the Screenshots smart
album membership, per-asset re-resolution by id, and a streaming download.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ic2x.config import Config

logger = logging.getLogger("ic2x.icloud_photos")


class ReauthRequired(RuntimeError):
    """iCloud session needs interactive 2FA. The loop catches this, fires
    cfg.reauth_notify_cmd, and stops cleanly instead of blocking on input()."""


class PyiCloudThrottled(RuntimeError):
    """Apple returned a rate-limit / temporary-unavailable response. Back off."""


@dataclass(frozen=True)
class AssetMeta:
    id: str               # PhotoAsset.id (stable CloudKit record name) — seen-set key
    created: datetime     # capture time, tz-aware UTC
    filename: str
    is_live: bool
    width: int | None
    height: int | None


def _classify(exc: Exception) -> Exception:
    """Map a pyicloud exception to ReauthRequired / PyiCloudThrottled / itself."""
    from pyicloud.exceptions import (
        PyiCloudAPIResponseException,
        PyiCloudFailedLoginException,
        PyiCloudServiceUnavailable,
    )

    if isinstance(exc, PyiCloudFailedLoginException):
        return ReauthRequired(f"login rejected: {exc}")
    if isinstance(exc, PyiCloudServiceUnavailable):
        return PyiCloudThrottled(str(exc))
    if isinstance(exc, PyiCloudAPIResponseException):
        text = str(exc).lower()
        if any(s in text for s in ("421", "450", "authentication required", "missing", "token")):
            return ReauthRequired(str(exc))
        if any(s in text for s in ("429", "503", "throttl", "unavailable", "rate")):
            return PyiCloudThrottled(str(exc))
    return exc


class ICloudPhotos:
    """Thin wrapper around a single authenticated pyicloud session."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._api: Any = None

    # ── auth ───────────────────────────────────────────────────────────────────

    def _inject_proxy(self) -> None:
        # requests reads these at call time; must be set before constructing the API.
        if self._cfg.proxy_http:
            os.environ.setdefault("HTTP_PROXY", self._cfg.proxy_http)
            os.environ.setdefault("http_proxy", self._cfg.proxy_http)
        if self._cfg.proxy_https:
            os.environ.setdefault("HTTPS_PROXY", self._cfg.proxy_https)
            os.environ.setdefault("https_proxy", self._cfg.proxy_https)

    def _build(self) -> Any:
        from pyicloud import PyiCloudService

        if self._cfg.icloud_with_family and not self._cfg.icloud_family_override:
            raise ReauthRequired(
                "ICLOUD_WITH_FAMILY=true mixes the shared family library into "
                "auto-posting. Set I_KNOW_FAMILY_PHOTOS_ARE_PUBLIC=true to allow it."
            )
        self._inject_proxy()
        try:
            return PyiCloudService(
                self._cfg.icloud_username,
                self._cfg.icloud_password,
                cookie_directory=str(self._cfg.icloud_cookie_dir),
                with_family=self._cfg.icloud_with_family,
                china_mainland=self._cfg.icloud_china_mainland,
            )
        except Exception as exc:  # noqa: BLE001 — re-raised as a typed error
            raise _classify(exc) from exc

    def ensure_session(self) -> None:
        """Construct from saved cookies, non-interactively. Raises ReauthRequired
        if 2FA/2SA is needed or the session is no longer trusted."""
        api = self._build()
        if api.requires_2fa or api.requires_2sa:
            raise ReauthRequired("interactive 2FA required — run `ic2x login`")
        self._api = api

    def interactive_login(self, code_provider) -> None:
        """For `ic2x login` only. code_provider() -> str prompts the user."""
        api = self._build()
        try:
            if api.requires_2fa:
                if not api.validate_2fa_code(code_provider()):
                    raise ReauthRequired("invalid 2FA code")
                if not api.is_trusted_session:
                    api.trust_session()
            elif api.requires_2sa:
                device = api.trusted_devices[0]
                api.send_verification_code(device)
                if not api.validate_verification_code(device, code_provider()):
                    raise ReauthRequired("invalid 2SA code")
        except ReauthRequired:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc
        self._api = api

    # ── photos ─────────────────────────────────────────────────────────────────

    @property
    def _photos(self) -> Any:
        if self._api is None:
            raise RuntimeError("ensure_session() not called")
        try:
            return self._api.photos
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc

    def _recent_album(self) -> Any:
        """Album iterating newest-ADDED first (genuinely new photos, taken or
        imported, sort first — unlike `.all`, which yields oldest capture date).
        Falls back to `.all` if the library accessor is unavailable."""
        photos = self._photos
        lib = getattr(photos, "_root_library", None)
        if lib is not None and hasattr(lib, "recently_added"):
            return lib.recently_added()
        return photos.all

    def iter_image_assets(self) -> Iterator[tuple[AssetMeta, Any]]:
        """Yield (metadata, live PhotoAsset) newest-ADDED-first, still images only.
        Videos/movies excluded by item_type. Metadata + a live downloadable asset;
        the asset is downloaded directly (never re-resolved by id — that hangs)."""
        try:
            for asset in self._recent_album():
                try:
                    if asset.item_type != "image":
                        continue
                    w, h = asset.dimensions or (None, None)
                    meta = AssetMeta(
                        id=asset.id, created=asset.created, filename=asset.filename,
                        is_live=bool(asset.is_live_photo), width=w, height=h,
                    )
                except Exception as exc:  # noqa: BLE001 — one bad asset is not fatal
                    logger.warning("icloud: skipping unreadable asset: %s", exc)
                    continue
                yield meta, asset
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc

    def screenshot_ids(self) -> set[str]:
        """Asset ids in the Apple Screenshots smart album (by id, not localized name)."""
        from pyicloud.services.photos_cloudkit import SmartAlbumEnum

        try:
            album = self._photos.albums[SmartAlbumEnum.SCREENSHOTS.value]
            return {a.id for a in album}
        except Exception as exc:  # noqa: BLE001
            logger.warning("icloud: Screenshots album unavailable (%s); "
                           "relying on the full-res EXIF gate only", exc)
            return set()

    def get_asset(self, asset_id: str) -> Any | None:
        """Re-resolve a live PhotoAsset by id (fresh signed download URLs)."""
        try:
            return self._photos.all.get(asset_id)
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc

    def download(self, asset: Any, version: str, dest: Path) -> Path:
        """Download `version` of `asset` to `dest`. Raises on failure / missing version."""
        try:
            data = asset.download(version)
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc
        if not data:
            raise RuntimeError(f"download: version {version!r} unavailable for {asset.id}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest
