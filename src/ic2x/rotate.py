"""
AI auto-rotation: the rotation model returns how to rotate (0/90/180/270° CW)
and we apply it locally. EXIF orientation is baked in first, so this only catches
what the camera tagged wrong.

  ic2x autorotate --count N   test it on N newest non-screenshot photos → rotate_out/
"""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from ic2x.config import Config, ensure_dirs, load_config
from ic2x.icloud_photos import ICloudPhotos, PyiCloudThrottled, ReauthRequired
from ic2x.judge_rotation import call_rotation
from ic2x.utils import ui
from ic2x.utils.ai_client import get_run_usage, parse_model_effort, reset_run_usage
from ic2x.utils.cost_report import compute_cost, format_total_cost_line
from ic2x.utils.image_utils import oriented

logger = logging.getLogger("ic2x.rotate")


def _bake_orientation(path: Path) -> None:
    """Apply EXIF orientation into pixels + strip EXIF, so the AI rotation operates
    on the same EXIF-upright image the model is shown."""
    with Image.open(path) as im:
        im = oriented(im)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.save(path, "JPEG", quality=92, exif=b"")


def autorotate_thumb(path: Path, cfg: Config) -> int:
    """Bake EXIF orientation, ask the rotation model, apply CW degrees in place.
    Returns the degrees applied (0 = already upright). Fail-open."""
    _bake_orientation(path)
    rot, _el, _used = call_rotation(path, cfg)
    deg = 0 if rot.get("upright") else int(rot.get("rotate_cw_degrees", 0) or 0)
    if deg in (90, 180, 270):
        from ic2x.bot import _apply_rotation
        _apply_rotation(path, deg)
        return deg
    return 0


def autorotate(count: int = 30) -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    from ic2x.utils.logging_setup import setup_logging
    setup_logging(cfg.logs_dir)
    from ic2x.bot import _safe_name, _unlink

    ic = ICloudPhotos(cfg)
    try:
        ic.ensure_session()
    except ReauthRequired as exc:
        ui.err(f"iCloud session needed — run `ic2x login`. ({exc})")
        return

    ss = ic.screenshot_ids()
    ui.info(f"Collecting {count} non-screenshot photos (skipping {len(ss)} screenshots) …")
    items = []
    try:
        for meta, asset in ic.iter_image_assets():
            if meta.id in ss:
                continue
            dest = cfg.work_dir / f"rot_{_safe_name(meta.id)}.jpg"
            try:
                ic.download(asset, cfg.thumb_version, dest)
            except (ReauthRequired, PyiCloudThrottled):
                raise
            except Exception:  # noqa: BLE001
                continue
            items.append((meta, dest))
            if len(items) >= count:
                break
    except (ReauthRequired, PyiCloudThrottled) as exc:
        ui.err(f"iCloud error: {exc}")
        return
    if not items:
        ui.warn("No non-screenshot photos found.")
        return

    ui.info(f"Asking {cfg.rotation_model} which way to rotate {len(items)} images (parallel) …")
    reset_run_usage()

    def _rot(it):
        meta, thumb = it
        return meta, thumb, autorotate_thumb(thumb, cfg)

    with ThreadPoolExecutor(max_workers=min(16, len(items))) as ex:
        results = list(ex.map(_rot, items))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = cfg.work_dir.parent / "rotate_out" / ts
    out.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for meta, thumb, deg in results:
        counts[deg] += 1
        stem = meta.filename.rsplit(".", 1)[0]
        tag = "upright" if deg == 0 else f"rot{deg}"
        shutil.copy(str(thumb), str(out / f"{tag}__{stem}.jpg"))
        _unlink(thumb)

    total, _bd = compute_cost(get_run_usage())
    rotated = sum(v for k, v in counts.items() if k)
    print("\n=== rotation summary ===")
    for deg in (0, 90, 180, 270):
        if counts.get(deg):
            print(f"  {('upright (0°)' if deg == 0 else f'{deg}° CW'):14}: {counts[deg]}")
    print(f"\n{rotated} of {len(results)} flagged for rotation   |   {format_total_cost_line(total)}")
    ui.ok(f"rotated thumbnails → {out}")
    ui.info("open it — every image should look upright; the 'rot90/180/270' names "
            "are the corrections to verify.")
