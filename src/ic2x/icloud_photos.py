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
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from ic2x.config import Config

logger = logging.getLogger("ic2x.icloud_photos")

# Added − captured beyond this = a bulk-imported archive photo (chronologically
# misplaced in rank order); generous margin for late phone syncs.
_ARCHIVE_IMPORT_GAP = timedelta(days=30)


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


_ICLOUD_RETRY_ATTEMPTS = 3        # one try + two retries before giving up
# A probe miss is evidence of deletion, not proof — it also happens when the rank
# baseline and the learned drift disagree, which never self-corrects because drift
# is only refreshed by a hit. Real deletions come in handfuls; a long unbroken run
# means the probe is wrong. Cap enforced in _fetch_by_rank._drop (2026-08-11: an
# uncapped run dropped 67139 rows and emptied the library index).
_MAX_CONSECUTIVE_DROPS = 25

_ICLOUD_RETRY_DELAY = 2.0         # seconds — close succession, for a transient iCloud blip


def _icloud_retry(fn, label: str):
    """Run `fn`, retrying a transient 'Request failed to iCloud' blip a few times in close
    succession. Reauth (needs 2FA) and throttling are re-raised AT ONCE so the bot loop's
    own handlers take over — only generic request failures are worth a quick retry."""
    for attempt in range(1, _ICLOUD_RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            classified = _classify(exc)
            if isinstance(classified, (ReauthRequired, PyiCloudThrottled)):
                raise classified from exc
            if attempt == _ICLOUD_RETRY_ATTEMPTS:
                raise classified from exc
            logger.warning("icloud: %s failed (%d/%d) — retrying in %.0fs … (%s)",
                           label, attempt, _ICLOUD_RETRY_ATTEMPTS, _ICLOUD_RETRY_DELAY,
                           str(exc)[:100])
            time.sleep(_ICLOUD_RETRY_DELAY)


class ICloudPhotos:
    """Thin wrapper around a single authenticated pyicloud session."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._api: Any = None
        self._ss_cache: set[str] | None = None  # Screenshots-album membership cache
        self._ss_cache_age = 0                   # cycles since last refresh

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
        # Validate config BEFORE importing the heavy pyicloud lib, so a clear
        # creds/family error surfaces even where pyicloud isn't importable.
        if not self._cfg.icloud_username or not self._cfg.icloud_password:
            raise ReauthRequired(
                "iCloud credentials missing — set ICLOUD_USERNAME and ICLOUD_PASSWORD "
                "in .env (see .env.example)."
            )
        if self._cfg.icloud_with_family and not self._cfg.icloud_family_override:
            raise ReauthRequired(
                "ICLOUD_WITH_FAMILY=true mixes the shared family library into "
                "auto-posting. Set I_KNOW_FAMILY_PHOTOS_ARE_PUBLIC=true to allow it."
            )
        from pyicloud import PyiCloudService

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
        self._enforce_session_timeout(api)
        self._api = api

    @staticmethod
    def _enforce_session_timeout(api: Any, seconds: int = 90) -> None:
        """pyicloud issues requests with NO timeout, and urllib3 sets
        sock.settimeout(None) on reads when none is given — so a dead HTTPS
        connection blocks an SSL read FOREVER and freezes the whole bot mid-scan
        (observed twice, 2026-07-14: 0%% CPU, stuck in poll()). A global
        socket.setdefaulttimeout does NOT cover this (urllib3 overrides it).
        Inject a default per-request timeout at the requests-session level, so a
        stalled call raises within `seconds` and flows into the existing
        transient-retry handling."""
        try:
            session = api.session
            orig = session.request

            def request_with_timeout(method, url, **kw):
                kw.setdefault("timeout", seconds)
                return orig(method, url, **kw)

            session.request = request_with_timeout
        except Exception as exc:  # noqa: BLE001 — never block startup on this guard
            logger.warning("icloud: could not enforce session timeout: %s", exc)

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
            # No session object means authentication has not succeeded — which IS a
            # ReauthRequired, and must be typed as one. As a bare RuntimeError it
            # slipped past the loop's reauth handling into the generic error path,
            # counting toward the errors>=6 re-exec instead of the notify-and-poll
            # that recovers when `ic2x login` lands. On 2026-09-06 a failed
            # ensure_session() left this None and the bot restarted 14 times.
            raise ReauthRequired("iCloud session not established — run `ic2x login`")
        try:
            return self._api.photos
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc

    def _recent_album(self) -> Any:
        """Album iterating newest-ADDED first (genuinely new photos, taken or
        imported, sort first). Falls back to `.all` if the library accessor is
        unavailable."""
        photos = self._photos
        lib = getattr(photos, "_root_library", None)
        if lib is not None and hasattr(lib, "recently_added"):
            return lib.recently_added()
        return photos.all

    def _iter_all_backwards(self) -> Iterator[Any]:
        """The WHOLE library, newest CloudKit asset-date first — the back-in-time
        reservoir behind the small Recently-Added window. Ordering mirrors the
        Photos app's All Photos grid (bulk imports rank at their import time).
        pyicloud's own DESCENDING album path miscounts synthetic albums, so this
        pages the genuine ascending `.all` album from its tail, reversing each
        window. len() is read once; concurrent additions only shift ranks by a
        few slots, which the caller's id-dedup + DB seen-skip absorb."""
        from pyicloud.services.photos_cloudkit.constants import DirectionEnum

        photos = self._photos
        lib = getattr(photos, "_root_library", None)
        album = lib.all if lib is not None else photos.all
        page = 100
        offset = max(len(album) - page, 0)
        while True:
            window = list(album._get_photos_at(offset, DirectionEnum.ASCENDING, page))
            for photo in reversed(window):
                yield photo
            if offset == 0:
                break
            offset = max(offset - page, 0)

    @staticmethod
    def _is_archive_import(asset: Any) -> bool:
        """True when the asset was ADDED long after capture — a bulk-imported old
        archive photo. Rank order places imports at their IMPORT time, which broke
        chronology (a 2026 walk-back jumped straight to a 2004 ski archive, user
        report 2026-07-14). Organic photos (added ≈ captured) already flow in true
        reverse-chronological order, so archive imports are deferred to a final
        phase — which is also where they belong chronologically (oldest era)."""
        try:
            created, added = asset.created, asset.added_date
            if created is None or added is None:
                return False
            return (added - created) > _ARCHIVE_IMPORT_GAP
        except Exception:  # noqa: BLE001 — missing/odd fields → treat as organic
            return False

    # ── Capture-date catalog: build / refresh / positional fetch ────────────────

    def _all_album(self) -> Any:
        photos = self._photos
        lib = getattr(photos, "_root_library", None)
        return lib.all if lib is not None else photos.all

    def ensure_catalog(self, db: Any, page: int = 200) -> None:
        """Build the capture-date catalog once (metadata sweep of the whole
        library, ~10 min for 67k assets), then keep it fresh each call by walking
        pages forward from the album HEAD until a whole page is already known.

        The .all album is NEWEST-FIRST (offset 0 = most recent — verified live
        2026-08-04: offset 0 was that morning's photo, the tail was from 2004), the
        same ordering _fetch_by_rank relies on. This refresh used to scan from the
        TAIL on the belief that new assets land there; the tail is the 2004 photos,
        so every refresh hit an all-known page and broke immediately. The catalog
        silently stopped growing after the initial sweep — by the time it was caught
        the library had 69974 assets against 67741 cataloged, and the walk-back
        could not see ANY photo from the preceding three weeks."""
        from pyicloud.services.photos_cloudkit.constants import DirectionEnum

        album = self._all_album()
        total = len(album)

        def _rows(offset: int) -> list[tuple[str, str | None, int]]:
            out = []
            for i, a in enumerate(album._get_photos_at(offset, DirectionEnum.ASCENDING, page)):
                try:
                    created = a.created.isoformat() if a.created else None
                except Exception:  # noqa: BLE001
                    created = None
                out.append((a.id, created, offset + i))
            return out

        if db.get_state("catalog_complete") != "1":
            start = db.catalog_count()  # resume an interrupted build
            logger.info("icloud: cataloging library capture dates — %d/%d done, "
                        "sweeping the remaining metadata (this is a one-time pass)",
                        start, total)
            offset = start
            while offset < total:
                rows = _rows(offset)
                if not rows:
                    break
                db.catalog_upsert_many(rows)
                offset += len(rows)
                if offset % 2000 < page:
                    logger.info("icloud: catalog %d/%d …", offset, total)
            db.set_state("catalog_complete", "1")
            logger.info("icloud: catalog complete — %d assets indexed", db.catalog_count())
            return

        # ── incremental HEAD refresh ──
        # Collect every unknown asset from offset 0 forward, stopping at the first
        # page that is already fully cataloged.
        #
        # Stored ranks are a FROZEN baseline from the initial sweep, and _fetch_by_rank
        # absorbs everything that has moved since with one learned `drift`. So the new
        # rows must join that same baseline, NOT the live album's offsets: an asset
        # added after the sweep sits BEFORE the old rank 0, hence a negative rank.
        # Rewriting existing ranks to match the live album instead (a global shift, as
        # this did until 2026-08-13) is what broke it — with both the stored rank and
        # the drift moving, probes missed, and a miss is read as "deleted from iCloud".
        # `drift` only updates on a HIT, so the first miss froze it at 0 and every
        # later probe missed too: 67139 catalog rows were dropped in a single run.
        offset = 0
        fresh: list[tuple[str, str | None, int]] = []
        while offset < total:
            rows = _rows(offset)
            if not rows:
                break
            known = db.catalog_known([r[0] for r in rows])
            new = [r for r in rows if r[0] not in known]
            fresh.extend(new)
            if len(new) < len(rows):   # reached already-cataloged territory
                break
            # Advance by what iCloud ACTUALLY returned, never by the requested page:
            # it silently CLAMPS a page request to ~49-100 (the initial sweep has
            # always advanced by len(rows) for this reason). Stepping by `page`
            # skipped every asset past the short page, leaving the catalog full of
            # holes — measured 2026-09-07: offset 0 needed drift 1395 while offset
            # 3000+ needed 2697, and offset 500 was not cataloged at all. One global
            # drift cannot span two regimes, so every probe missed.
            offset += len(rows)
        if fresh:
            # Re-base onto the frozen scale, CONTINUING below whatever the previous
            # refreshes already used. Numbering each batch -n..-1 in isolation was
            # wrong: every refresh restarted at -1, so batches collided and sat at the
            # wrong offsets. Observed 2026-09-07 — 1420 assets sharing 391 slots, every
            # probe missing, and (because drift only updates on a HIT) the walk-back
            # unable to resolve a single photo for 57 hours.
            #
            # One global drift only fits if rank is a position on ONE continuous scale:
            # an old asset sits at rank + A and a newer one at rank - k, where A is
            # everything added since the sweep. So each batch extends the run downward.
            n = len(fresh)
            base = min(0, db.catalog_min_rank())
            rebased = [(aid, created, base - n + i)
                       for i, (aid, created, _off) in enumerate(fresh)]
            db.catalog_upsert_many(rebased)
            logger.info("icloud: catalog refreshed — %d new assets indexed at ranks "
                        "%d..%d (existing ranks untouched)", n, base - n, base - 1)

    def _fetch_by_rank(self, asset_id: str, rank: int, db: Any = None) -> Any | None:
        """Live PhotoAsset for a cataloged id: fetch a window around its recorded
        rank and match by id (positional — resolving by id hangs).

        Robustness (2026-07-15 field lessons):
        - The list is newest-first, so photos added since catalog time shift ALL
          old ranks uniformly — a global drift estimate, learned from each hit,
          keeps probes centered (rank + drift).
        - Consecutive chronological candidates cluster in rank; the window cache
          serves them without extra calls.
        - An id whose (drifted) rank the current window COVERS but doesn't contain
          was deleted from iCloud (user deleted a batch of Jul-9 photos) — skip
          for free and, when `db` is given, drop its catalog row so future cycles
          don't re-probe it."""
        from pyicloud.services.photos_cloudkit.constants import DirectionEnum

        drift = getattr(self, "_rank_drift", 0)
        cache: dict[str, Any] = getattr(self, "_fetch_cache", None) or {}
        ranks: dict[str, int] = getattr(self, "_fetch_ranks", None) or {}
        bounds = getattr(self, "_fetch_bounds", None)

        def _hit(a_id: str) -> Any:
            actual = ranks.get(a_id)
            if actual is not None:
                self._rank_drift = actual - rank
            self._drop_streak = 0        # probes are landing again
            return cache[a_id]

        def _drop(a_id: str, why: str) -> None:
            """Forget a cataloged asset that iCloud no longer returns — with a hard
            per-run cap.

            A miss is only EVIDENCE of deletion, never proof: it also happens whenever
            the rank baseline and the learned drift disagree. Since `drift` is refreshed
            only by a hit, a systematic mismatch never self-corrects, and unbounded
            trust in that inference cost the whole library index on 2026-08-11 — 67139
            rows dropped in one run because probing never recovered. Photos genuinely
            vanish a handful at a time, so a long unbroken run of misses means the
            PROBE is wrong, not the library. Stop dropping and let the next cycle
            rebuild its window from scratch."""
            streak = getattr(self, "_drop_streak", 0) + 1
            self._drop_streak = streak
            if streak > _MAX_CONSECUTIVE_DROPS:
                if streak == _MAX_CONSECUTIVE_DROPS + 1:
                    logger.error(
                        "icloud: %d cataloged assets in a row looked deleted — treating "
                        "this as a PROBE failure, not %d real deletions, and keeping the "
                        "catalog. Ranks and the live album have diverged; the next cycle "
                        "re-probes with a fresh window.", streak, streak)
                return
            logger.warning("icloud: cataloged asset %s %s — dropping", a_id[:8], why)
            db.catalog_delete(a_id)

        if asset_id in cache:
            return _hit(asset_id)
        expected = rank + drift
        if bounds and bounds[0] <= expected <= bounds[1]:
            # window covers where it should be, id absent → looks deleted
            if db is not None:
                _drop(asset_id, "absent from covered window")
            return None

        album = self._all_album()
        for width in (120, 700):
            start = max(expected - width // 2, 0)
            # iCloud pages at ~100 per query and silently CLAMPS larger requests
            # (a "700-wide" single fetch returned 100 and ate valid catalog rows,
            # 2026-07-16) — accumulate explicit 100-page chunks instead.
            window_list: list[Any] = []
            off = start
            while off < start + width:
                page = list(album._get_photos_at(off, DirectionEnum.ASCENDING,
                                                 min(100, start + width - off)))
                window_list.extend(page)
                if not page:
                    break  # truly no data at this offset
                off += len(page)  # iCloud returns SHORT pages (~49) mid-library — keep going
            cache = {a.id: a for a in window_list}
            ranks = {a.id: start + i for i, a in enumerate(window_list)}
            self._fetch_cache, self._fetch_ranks = cache, ranks
            self._fetch_bounds = (start, start + len(window_list) - 1)
            if asset_id in cache:
                return _hit(asset_id)
        covered = (self._fetch_bounds[0] <= expected <= self._fetch_bounds[1]
                   and len(window_list) > 0)
        if covered and db is not None:
            _drop(asset_id, f"not at rank {rank}(+{drift} drift) though window covered it")
        else:
            logger.warning("icloud: cataloged asset %s not found near rank %d(+%d drift) — "
                           "window did NOT cover it; keeping for retry", asset_id[:8],
                           rank, drift)
        return None

    def iter_image_assets_chrono(self, db: Any, min_score: int | None = None
                                 ) -> Iterator[tuple[AssetMeta, Any]]:
        """The bot's walk-back order: Recently-Added first (fresh shots; bulk
        imports deferred — the catalog places them at their true position), then
        every not-yet-decided asset in strict capture-date-DESC order from the
        persisted catalog. Never starves while unseen photos exist; never jumps
        eras (user steer 2026-07-14: 'go back in time chronologically')."""
        yielded: set[str] = set()

        def _meta(asset: Any) -> AssetMeta | None:
            try:
                if asset.item_type != "image":
                    return None
                w, h = asset.dimensions or (None, None)
                return AssetMeta(id=asset.id, created=asset.created,
                                 filename=asset.filename,
                                 is_live=bool(asset.is_live_photo), width=w, height=h)
            except Exception as exc:  # noqa: BLE001
                logger.warning("icloud: skipping unreadable asset: %s", exc)
                return None

        try:
            self.ensure_catalog(db)
            try:
                for asset in self._recent_album():
                    if getattr(asset, "id", None) in yielded or self._is_archive_import(asset):
                        continue
                    m = _meta(asset)
                    if m is not None:
                        yielded.add(m.id)
                        yield m, asset
            except Exception as exc:  # noqa: BLE001 — recent-album query is flaky
                # (observed 2026-07-15: the ADDED-index endpoint failing while the
                # catalog's queries work). The tail-refresh above already cataloged
                # anything new, so the catalog phase still serves newest-first —
                # losing only the added-order preference, never any photo.
                logger.warning("icloud: recent-window query failed (%s) — falling "
                               "through to the catalog walk", str(exc)[:120])
            logger.info("icloud: recent window done (%d yielded) — continuing back "
                        "in time chronologically via the catalog", len(yielded))
            for row in db.catalog_unseen_desc(min_score):
                if row["asset_id"] in yielded:
                    continue
                asset = self._fetch_by_rank(row["asset_id"], row["rank"], db)
                if asset is None:
                    continue
                m = _meta(asset)
                if m is not None:
                    yielded.add(m.id)
                    yield m, asset
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc

    def iter_image_assets(self) -> Iterator[tuple[AssetMeta, Any]]:
        """Yield (metadata, live PhotoAsset), still images only, in CHRONOLOGICAL
        walk-back order:
          1. Recently-Added window (newest first) — new shots always come first;
          2. the whole library newest-capture-first (organic photos, whose rank
             order ≈ capture order), skipping bulk-imported archives;
          3. only when the organic timeline is exhausted: the deferred archive
             imports (chronologically the oldest era anyway).
        The caller's DB seen-skip decides how deep each cycle actually walks; this
        iterator never starves while unseen photos exist anywhere in the library.
        Videos/movies excluded by item_type. Metadata + a live downloadable asset;
        the asset is downloaded directly (never re-resolved by id — that hangs)."""
        yielded: set[str] = set()
        deferred = 0

        def _images(assets: Iterator[Any], archive_pass: bool
                    ) -> Iterator[tuple[AssetMeta, Any]]:
            nonlocal deferred
            for asset in assets:
                try:
                    if asset.item_type != "image" or asset.id in yielded:
                        continue
                    if not archive_pass and self._is_archive_import(asset):
                        deferred += 1
                        continue
                    w, h = asset.dimensions or (None, None)
                    meta = AssetMeta(
                        id=asset.id, created=asset.created, filename=asset.filename,
                        is_live=bool(asset.is_live_photo), width=w, height=h,
                    )
                except Exception as exc:  # noqa: BLE001 — one bad asset is not fatal
                    logger.warning("icloud: skipping unreadable asset: %s", exc)
                    continue
                yielded.add(asset.id)
                yield meta, asset

        try:
            yield from _images(iter(self._recent_album()), False)
            logger.info("icloud: recent window exhausted — walking back in time "
                        "chronologically (%d yielded, %d archive imports deferred)",
                        len(yielded), deferred)
            yield from _images(self._iter_all_backwards(), False)
            if deferred:
                logger.info("icloud: organic timeline exhausted — %d deferred "
                            "archive-import assets now eligible (oldest era)", deferred)
                yield from _images(self._iter_all_backwards(), True)
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc

    def screenshot_ids(self) -> set[str]:
        """Asset ids in the Apple Screenshots smart album (by id, not localized name)."""
        from pyicloud.services.photos_cloudkit import SmartAlbumEnum

        def _fetch() -> set[str]:
            album = self._photos.albums[SmartAlbumEnum.SCREENSHOTS.value]
            return {a.id for a in album}

        try:
            return _icloud_retry(_fetch, "Screenshots album")
        except Exception as exc:  # noqa: BLE001
            logger.warning("icloud: Screenshots album unavailable (%s); "
                           "relying on the full-res EXIF gate only", exc)
            return set()

    def screenshot_ids_cached(self, refresh_every: int) -> set[str]:
        """Screenshots-album membership, re-fetched from iCloud only once every
        `refresh_every` cycles (≥1); the cached set is returned in between. Iterating
        the whole smart album is a real per-cycle iCloud round-trip, and the album
        barely changes minute-to-minute, so caching cuts API load + throttling risk.
        Safe staleness: a brand-new screenshot that slips past a stale set is still
        rejected by the fail-closed EXIF gate on the winner's full-res original."""
        if self._ss_cache is None or self._ss_cache_age >= max(1, refresh_every):
            self._ss_cache = self.screenshot_ids()
            self._ss_cache_age = 0
        self._ss_cache_age += 1
        return self._ss_cache

    def get_asset(self, asset_id: str) -> Any | None:
        """Re-resolve a live PhotoAsset by id (fresh signed download URLs)."""
        try:
            return self._photos.all.get(asset_id)
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc

    def download(self, asset: Any, version: str, dest: Path) -> Path:
        """Download `version` of `asset` to `dest`. Raises on failure / missing version.
        Transient iCloud request failures are retried in close succession first."""
        data = _icloud_retry(lambda: asset.download(version), f"download({version})")
        if not data:
            raise RuntimeError(f"download: version {version!r} unavailable for {asset.id}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest
