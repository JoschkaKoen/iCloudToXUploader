"""
AI auto-rotation (pick-4): the judge sees all 4 rotations of the photo in one
multi-image call and picks the upright one; a second 2-image call confirms any
proposed fix against the as-shot version (straight-down shots stay as shot).
We apply the resulting 0/90/180/270° CW locally. EXIF orientation is baked in
first; this catches photos whose camera metadata is wrong (EXIF says "upright"
but the content is sideways).

  ic2x autorotate --count N [--models "qwen3.5-flash, 2000; gemini-2.5-flash-lite"]
    → rotate_out/<ts>/<model>/  (one folder per model, for visual comparison)
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
from ic2x.utils.ai_client import get_run_usage, reset_run_usage
from ic2x.utils.cost_report import compute_cost, format_total_cost_line
from ic2x.utils.image_utils import oriented

logger = logging.getLogger("ic2x.rotate")


def _bake_orientation(path: Path) -> None:
    """Apply EXIF orientation into pixels + strip EXIF."""
    with Image.open(path) as im:
        im = oriented(im)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.save(path, "JPEG", quality=92, exif=b"")


def _ai_rotate(path: Path, cfg: Config, model_string: str | None = None) -> int:
    """AI rotation only (orientation assumed already baked). Returns CW degrees applied."""
    rot, _el, _used = call_rotation(path, cfg, model_string=model_string)
    deg = 0 if rot.get("upright") else int(rot.get("rotate_cw_degrees", 0) or 0)
    if deg in (90, 180, 270):
        from ic2x.bot import _apply_rotation
        _apply_rotation(path, deg)
        return deg
    return 0


def autorotate_thumb(path: Path, cfg: Config, model_string: str | None = None) -> int:
    """Bake EXIF orientation, then AI-rotate in place. Used by `classify --rotate`."""
    _bake_orientation(path)
    return _ai_rotate(path, cfg, model_string)


def _safe_model(m: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in m).strip("_")


def autorotate(count: int = 30, models: list[str] | None = None,
               version: str | None = None) -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    from ic2x.utils.logging_setup import setup_logging
    setup_logging(cfg.logs_dir)
    from ic2x.bot import _safe_name, _unlink

    models = models or [cfg.rotation_model]
    ic = ICloudPhotos(cfg)
    try:
        ic.ensure_session()
    except ReauthRequired as exc:
        ui.err(f"iCloud session needed — run `ic2x login`. ({exc})")
        return

    ss = ic.screenshot_ids()
    ui.info(f"Collecting {count} non-screenshot photos (skipping {len(ss)} screenshots) …")
    bases = []  # (meta, pristine_baked_path)
    try:
        for meta, asset in ic.iter_image_assets():
            if meta.id in ss:
                continue
            dest = cfg.work_dir / f"base_{_safe_name(meta.id)}.jpg"
            try:
                ic.download(asset, version or cfg.thumb_version, dest)
            except (ReauthRequired, PyiCloudThrottled):
                raise
            except Exception:  # noqa: BLE001
                continue
            _bake_orientation(dest)  # EXIF first, once
            bases.append((meta, dest))
            if len(bases) >= count:
                break
    except (ReauthRequired, PyiCloudThrottled) as exc:
        ui.err(f"iCloud error: {exc}")
        return
    if not bases:
        ui.warn("No non-screenshot photos found.")
        return

    ui.info(f"Comparing rotation models {models} on {len(bases)} images (all parallel) …")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = cfg.work_dir.parent / "rotate_out" / ts
    reset_run_usage()

    tasks = [(mi, model, meta, base)
             for mi, model in enumerate(models) for (meta, base) in bases]

    def _rot(task):
        mi, model, meta, base = task
        tmp = cfg.work_dir / f"r{mi}_{_safe_name(meta.id)}.jpg"
        shutil.copy(str(base), str(tmp))
        deg = _ai_rotate(tmp, cfg, model_string=model)
        return mi, model, meta, tmp, deg

    with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as ex:
        results = list(ex.map(_rot, tasks))

    per_model: dict[str, Counter] = {m: Counter() for m in models}
    for mi, model, meta, tmp, deg in results:
        per_model[model][deg] += 1
        mdir = out / f"{mi}_{_safe_model(model)}"
        mdir.mkdir(parents=True, exist_ok=True)
        stem = meta.filename.rsplit(".", 1)[0]
        tag = "upright" if deg == 0 else f"rot{deg}"
        shutil.copy(str(tmp), str(mdir / f"{tag}__{stem}.jpg"))
        _unlink(tmp)
    for _meta, base in bases:
        _unlink(base)

    total, _bd = compute_cost(get_run_usage())
    print("\n=== rotation model comparison ===")
    for model in models:
        c = per_model[model]
        rotated = sum(v for k, v in c.items() if k)
        breakdown = ", ".join(f"{k}°×{v}" for k, v in sorted(c.items()) if k) or "none"
        print(f"  {model:26} flagged {rotated}/{len(bases)} for rotation  ({breakdown})")
    print(f"\ntotal {format_total_cost_line(total)}")
    ui.ok(f"per-model folders → {out}")
    ui.info("open each model's folder; compare which one rotates the problem photos "
            "(e.g. IMG_7674) correctly upright.")
