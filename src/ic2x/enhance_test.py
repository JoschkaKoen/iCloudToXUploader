"""
ic2x enhance-test — try Aliyun VIAPI image enhancement on real photos and write
labeled before/after montages, so you can judge quality BEFORE wiring it into the
bot. Mirrors `autorotate` (rotate.py).

  ic2x enhance-test [--count N] [--dir FOLDER]
                    [--capabilities superres,color] [--max-edge 1920]
    → enhance_out/<ts>/   <NN>_<stem>__compare.jpg  (original | each capability)
      plus the individual JPEGs and summary.md.

Posts nothing, but makes REAL Aliyun calls (0.02 RMB each after 100/month free).
Inputs are downscaled to fit the API limits (≤1920×1080, <3 MB). With UpscaleFactor=1
the output keeps the input size, so before/after compare at the same resolution.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from ic2x.config import ensure_dirs, load_config
from ic2x.icloud_photos import ICloudPhotos, PyiCloudThrottled, ReauthRequired
from ic2x.utils import ui
from ic2x.utils.aliyun_viapi import (
    CAPABILITIES, MAX_LONG_EDGE, MAX_SHORT_EDGE, ViapiError, check_credentials, fit_jpeg,
)

logger = logging.getLogger("ic2x.enhance_test")

_MAX_WORKERS = 2  # VIAPI QPS = 2
_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".webp")


def _prep_input(src: Path, dest: Path, max_long: int, max_short: int) -> tuple[int, int]:
    """Orient, downscale to fit (max_long × max_short), re-encode JPEG under the API
    size cap. Delegates to the shared helper the bot's posting path also uses."""
    return fit_jpeg(src, dest, max_long=max_long, max_short=max_short)


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


def _collect_from_dir(folder: str, count: int, out: Path, max_long: int, max_short: int
                      ) -> list[tuple[str, Path]]:
    src_dir = Path(folder).expanduser()
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in _EXTS)[:count]
    if not files:
        ui.err(f"no images ({'/'.join(_EXTS)}) in {src_dir}")
        return []
    ui.info(f"Preparing {len(files)} images from {src_dir} …")
    inputs = []
    for i, src in enumerate(files):
        prepped = out / f"{i:02d}_{src.stem}__0_original.jpg"
        try:
            _prep_input(src, prepped, max_long, max_short)
            inputs.append((src.stem, prepped))
        except Exception as exc:  # noqa: BLE001
            ui.warn(f"skip {src.name}: {exc}")
    return inputs


def _collect_from_icloud(cfg, count: int, out: Path, max_long: int, max_short: int
                         ) -> list[tuple[str, Path]]:
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
            raw = cfg.work_dir / f"enh_{_safe_name(meta.id)}"
            try:
                ic.download(asset, "original", raw)
            except (ReauthRequired, PyiCloudThrottled):
                raise
            except Exception:  # noqa: BLE001
                continue
            stem = meta.filename.rsplit(".", 1)[0]
            prepped = out / f"{len(inputs):02d}_{stem}__0_original.jpg"
            try:
                _prep_input(raw, prepped, max_long, max_short)
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


def enhance_test(count: int = 8, capabilities: tuple[str, ...] = ("superres", "color"),
                 folder: str | None = None, max_edge: int = MAX_LONG_EDGE) -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    from ic2x.utils.logging_setup import setup_logging
    setup_logging(cfg.logs_dir)

    caps = [c for c in capabilities if c in CAPABILITIES]
    unknown = [c for c in capabilities if c not in CAPABILITIES]
    if unknown:
        ui.warn(f"unknown capabilities ignored: {', '.join(unknown)} "
                f"(available: {', '.join(CAPABILITIES)})")
    if not caps:
        ui.err(f"no valid capabilities. Available: {', '.join(CAPABILITIES)}")
        return
    try:
        check_credentials()
    except ViapiError as exc:
        ui.err(str(exc))
        return

    max_short = round(max_edge * MAX_SHORT_EDGE / MAX_LONG_EDGE)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = cfg.work_dir.parent / "enhance_out" / ts
    out.mkdir(parents=True, exist_ok=True)

    inputs = (_collect_from_dir(folder, count, out, max_edge, max_short) if folder
              else _collect_from_icloud(cfg, count, out, max_edge, max_short))
    if not inputs:
        ui.warn("No images to enhance.")
        return

    ui.info(f"Enhancing {len(inputs)} images × {len(caps)} capabilities "
            f"({', '.join(caps)}) via Aliyun VIAPI …")
    tasks = [(idx, label, prepped, cap)
             for idx, (label, prepped) in enumerate(inputs) for cap in caps]
    results: dict[tuple[int, str], Path] = {}
    stats = {c: {"ok": 0, "fail": 0} for c in caps}

    def _do(task):
        idx, label, prepped, cap = task
        t0 = time.monotonic()
        try:
            data = CAPABILITIES[cap](prepped)
            dest = out / f"{idx:02d}_{label}__{caps.index(cap) + 1}_{cap}.jpg"
            dest.write_bytes(data)
            return task, dest, None, time.monotonic() - t0
        except Exception as exc:  # noqa: BLE001
            return task, None, str(exc), time.monotonic() - t0

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for task, dest, err, dt in ex.map(_do, tasks):
            idx, label, _prepped, cap = task
            if dest:
                results[(idx, cap)] = dest
                stats[cap]["ok"] += 1
                ui.info(f"   ✓ {label} · {cap} ({dt:.1f}s)")
            else:
                stats[cap]["fail"] += 1
                ui.warn(f"   ✗ {label} · {cap}: {err}")
    wall = time.monotonic() - t_start

    for idx, (label, prepped) in enumerate(inputs):
        panels = [("original", prepped)]
        panels += [(cap, results[(idx, cap)]) for cap in caps if (idx, cap) in results]
        try:
            _montage(panels, out / f"{idx:02d}_{label}__compare.jpg")
        except Exception as exc:  # noqa: BLE001
            logger.warning("montage failed for %s: %s", label, exc)

    n_calls = sum(s["ok"] + s["fail"] for s in stats.values())
    lines = [f"# enhance-test {ts}", "",
             f"images: {len(inputs)}   capabilities: {', '.join(caps)}   "
             f"calls: {n_calls}   wall: {wall:.1f}s", ""]
    lines += [f"- {c}: {stats[c]['ok']} ok, {stats[c]['fail']} fail" for c in caps]
    lines += ["", f"cost: first 100 calls/month free, then 0.02 RMB/call "
                  f"→ ≤ {n_calls * 0.02:.2f} RMB if the free quota is exhausted", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== enhance-test ===")
    for c in caps:
        print(f"  {c:14} {stats[c]['ok']} ok / {stats[c]['fail']} fail")
    print(f"  {n_calls} calls · {wall:.1f}s · ≤{n_calls * 0.02:.2f} RMB")
    ui.ok(f"open the __compare.jpg montages → {out}")
