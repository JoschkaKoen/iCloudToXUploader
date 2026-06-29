"""
ic2x polish-test — preview the free local adaptive polish on real photos and write
labeled before/after montages, so you can judge it BEFORE setting POLISH_ENABLED.
No API, no cost, no posting — purely local CPU work. Mirrors `enhance-test`.

  ic2x polish-test [--dir FOLDER] [--count N] [--intensities natural,punchy]
    → polish_out/<ts>/  <NN>_<stem>__compare.jpg  (original | each intensity)
      plus the individual JPEGs and summary.md.

With --dir it is fully offline. Without it, the iCloud newest N photos are used
(downloads originals only — still no AI / enhancement cost).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from ic2x import polish as polish_mod
from ic2x.config import ensure_dirs, load_config
from ic2x.icloud_photos import ICloudPhotos, PyiCloudThrottled, ReauthRequired
from ic2x.utils import ui
from ic2x.utils.aliyun_viapi import fit_jpeg

logger = logging.getLogger("ic2x.polish_test")

_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".webp")
_PREVIEW_LONG = 1600   # downscale long edge for a fast, reasonable-size preview
_PREVIEW_SHORT = 1600


def _montage(panels: list[tuple[str, Path]], dest: Path, panel_h: int = 720) -> None:
    """Labeled side-by-side montage of (label, image) panels at a common height."""
    imgs = []
    for label, p in panels:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        imgs.append((label, im.resize((max(1, round(w * panel_h / h)), panel_h), Image.LANCZOS)))
    gap, head = 8, 28
    total_w = sum(im.width for _, im in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (total_w, panel_h + head), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    x = 0
    for label, im in imgs:
        canvas.paste(im, (x, head))
        d.text((x + 5, 8), label, fill=(245, 245, 245))
        x += im.width + gap
    canvas.save(dest, "JPEG", quality=90)


def _collect_from_dir(folder: str, count: int, out: Path) -> list[tuple[str, Path]]:
    src_dir = Path(folder).expanduser()
    if not src_dir.is_dir():
        ui.err(f"not a folder: {src_dir}")
        return []
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in _EXTS)[:count]
    if not files:
        ui.err(f"no images ({'/'.join(_EXTS)}) in {src_dir}")
        return []
    ui.info(f"Preparing {len(files)} images from {src_dir} …")
    inputs = []
    for i, src in enumerate(files):
        prepped = out / f"{i:02d}_{src.stem}__0_original.jpg"
        try:
            fit_jpeg(src, prepped, max_long=_PREVIEW_LONG, max_short=_PREVIEW_SHORT)
            inputs.append((src.stem, prepped))
        except Exception as exc:  # noqa: BLE001
            ui.warn(f"skip {src.name}: {exc}")
    return inputs


def _collect_from_icloud(cfg, count: int, out: Path) -> list[tuple[str, Path]]:
    from ic2x.bot import _safe_name, _unlink

    ic = ICloudPhotos(cfg)
    try:
        ic.ensure_session()
    except ReauthRequired as exc:
        ui.err(f"iCloud session needed — run `ic2x login`. ({exc})")
        return []
    ss = ic.screenshot_ids()
    ui.info(f"Collecting {count} non-screenshot photos (skipping {len(ss)} screenshots) …")
    inputs: list[tuple[str, Path]] = []
    try:
        for meta, asset in ic.iter_image_assets():
            if meta.id in ss:
                continue
            raw = cfg.work_dir / f"pol_{_safe_name(meta.id)}"
            try:
                ic.download(asset, "original", raw)
            except (ReauthRequired, PyiCloudThrottled):
                raise
            except Exception:  # noqa: BLE001
                continue
            stem = meta.filename.rsplit(".", 1)[0]
            prepped = out / f"{len(inputs):02d}_{stem}__0_original.jpg"
            try:
                fit_jpeg(raw, prepped, max_long=_PREVIEW_LONG, max_short=_PREVIEW_SHORT)
                inputs.append((stem, prepped))
            except Exception as exc:  # noqa: BLE001
                ui.warn(f"skip {stem}: {exc}")
            finally:
                _unlink(raw)
            if len(inputs) >= count:
                break
    except (ReauthRequired, PyiCloudThrottled) as exc:
        ui.err(f"iCloud error: {exc}")
    return inputs


def polish_test(count: int = 8, folder: str | None = None,
                intensities: tuple[str, ...] = ("natural", "punchy")) -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    from ic2x.utils.logging_setup import setup_logging
    setup_logging(cfg.logs_dir)

    valid = tuple(i for i in intensities if i in polish_mod._PRESETS)
    if not valid:
        ui.err(f"no valid intensities. Available: {', '.join(polish_mod._PRESETS)}")
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = cfg.work_dir.parent / "polish_out" / ts
    out.mkdir(parents=True, exist_ok=True)

    inputs = (_collect_from_dir(folder, count, out) if folder
              else _collect_from_icloud(cfg, count, out))
    if not inputs:
        ui.warn("No images to polish.")
        return

    ui.info(f"Polishing {len(inputs)} images × {len(valid)} intensities "
            f"({', '.join(valid)}) — free, local, CPU-only …")
    n_changed = 0
    for idx, (label, prepped) in enumerate(inputs):
        panels = [("original", prepped)]
        for it in valid:
            it_cfg = replace(cfg, polish_enabled=True, polish_intensity=it)
            dest = out / f"{idx:02d}_{label}__{it}.jpg"
            try:
                with Image.open(prepped) as im:
                    polished = polish_mod.polish(im, it_cfg)
                    changed = polished is not im
                    polished.save(dest, "JPEG", quality=92)
                if changed:
                    n_changed += 1
                panels.append((it, dest))
            except Exception as exc:  # noqa: BLE001
                ui.warn(f"   ✗ {label} · {it}: {exc}")
        try:
            _montage(panels, out / f"{idx:02d}_{label}__compare.jpg")
        except Exception as exc:  # noqa: BLE001
            logger.warning("montage failed for %s: %s", label, exc)
        ui.info(f"   ✓ {label}")

    lines = [f"# polish-test {ts}", "",
             f"images: {len(inputs)}   intensities: {', '.join(valid)}   cost: 0 (local)", "",
             "Open the __compare.jpg montages: original | "
             + " | ".join(valid) + ".", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== polish-test ===")
    print(f"  {len(inputs)} images × {len(valid)} intensities · 0 RMB (fully local)")
    ui.ok(f"open the __compare.jpg montages → {out}")
