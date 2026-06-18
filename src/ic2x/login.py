"""
`ic2x login` — interactive iCloud sign-in. Establishes the pyicloud session
(2FA via the trusted device) into ICLOUD_COOKIE_DIR so the bot can run
non-interactively afterwards until the trust token expires (~weeks).

Also doubles as the live sanity check: prints the newest photo and the
Screenshots-album size to confirm photo access works on this account.
"""

from __future__ import annotations

from itertools import islice

from ic2x.config import ensure_dirs, load_config
from ic2x.icloud_photos import ICloudPhotos, ReauthRequired
from ic2x.utils import ui


def login() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    from ic2x.utils.logging_setup import setup_logging
    setup_logging(cfg.logs_dir)

    ic = ICloudPhotos(cfg)
    ui.info(f"Signing in to iCloud as {cfg.icloud_username} …")

    def code_provider() -> str:
        return input("Enter the 2FA code from your trusted device: ").strip()

    try:
        ic.interactive_login(code_provider)
    except ReauthRequired as exc:
        ui.err(f"login failed: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        ui.err(f"login error: {exc}")
        return

    ui.ok("iCloud session established.")

    # Live access check (the deferred Phase-0 facts).
    try:
        first = list(islice(ic.iter_image_assets(), 1))
        if first:
            meta, _ = first[0]
            ui.info(f"Newest photo: {meta.filename}  {meta.created.isoformat()}  "
                    f"{meta.width}x{meta.height}")
        ss = ic.screenshot_ids()
        ui.info(f"Screenshots album: {len(ss)} assets known")
    except Exception as exc:  # noqa: BLE001
        ui.warn(f"photo-access check failed (will retry at run time): {exc}")

    ui.ok("Ready. Run `ic2x bot` (keep X_DRY_RUN=true for burn-in).")
