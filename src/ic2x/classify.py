"""
`ic2x classify --count N` — diagnostic: run the judge over the N newest
non-screenshot photos (each as an individual image), in parallel, and save every
thumbnail to classify_out/<ts>/ named by verdict so postable-vs-not can be
inspected locally. No posting, no DB changes. Delete the folder when done.
"""

from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ic2x.bot import _safe_name, _unlink
from ic2x.config import ensure_dirs, load_config
from ic2x.icloud_photos import ICloudPhotos, PyiCloudThrottled, ReauthRequired
from ic2x.judge_burst import judge_burst
from ic2x.utils import ui
from ic2x.utils.ai_client import parse_model_effort
from ic2x.utils.cost_report import compute_cost, format_total_cost_line

logger = logging.getLogger("ic2x.classify")


def _label(v: dict) -> tuple[str, str]:
    """(label, note) for a verdict."""
    if not v.get("safe"):
        return "REJECT-unsafe", ",".join(v.get("flags") or []) or "unsafe"
    if not v.get("interesting"):
        return "REJECT-boring", v.get("reason") or "not interesting"
    return "POSTABLE", v.get("caption") or ""


def _safe(text: str, n: int = 45) -> str:
    s = "".join(c if (c.isalnum() or c in " -") else "_" for c in (text or "")).strip()
    return "-".join(s.split())[:n] or "na"


def classify(count: int = 30, rotate: bool = False) -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    from ic2x.utils.logging_setup import setup_logging
    setup_logging(cfg.logs_dir)

    ic = ICloudPhotos(cfg)
    try:
        ic.ensure_session()
    except ReauthRequired as exc:
        ui.err(f"iCloud session needed — run `ic2x login`. ({exc})")
        return

    ss = ic.screenshot_ids()
    ui.info(f"Collecting {count} non-screenshot photos (skipping {len(ss)} screenshots) …")
    items = []  # (meta, thumb_path)
    try:
        for meta, asset in ic.iter_image_assets():
            if meta.id in ss:
                continue
            dest = cfg.work_dir / f"cls_{_safe_name(meta.id)}.jpg"
            try:
                ic.download(asset, cfg.thumb_version, dest)
            except (ReauthRequired, PyiCloudThrottled):
                raise
            except Exception:  # noqa: BLE001 — undecodable, skip
                continue
            if rotate:
                from ic2x.rotate import autorotate_thumb
                try:
                    autorotate_thumb(dest, cfg)
                except Exception as exc:  # noqa: BLE001 — never block classification
                    logger.warning("rotate: skipped %s (%s)", meta.filename, exc)
            items.append((meta, dest))
            if len(items) >= count:
                break
    except (ReauthRequired, PyiCloudThrottled) as exc:
        ui.err(f"iCloud error: {exc}")
        return

    if not items:
        ui.warn("No non-screenshot photos found.")
        return

    ui.info(f"Classifying {len(items)} images with {cfg.judge_model} (parallel) …")

    def _judge(it):
        meta, thumb = it
        usage: dict = {}
        v, _el, _net = judge_burst([thumb], cfg, usage_out=usage)
        return meta, thumb, v, usage

    with ThreadPoolExecutor(max_workers=min(16, len(items))) as ex:
        results = list(ex.map(_judge, items))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = cfg.work_dir.parent / "classify_out" / ts
    out.mkdir(parents=True, exist_ok=True)

    agg = {"input": 0, "output": 0, "thinking": 0}
    rows = []
    for meta, thumb, v, usage in results:
        agg["input"] += usage.get("input", 0)
        agg["output"] += usage.get("output", 0)
        label, note = _label(v)
        shutil.copy(str(thumb), str(out / f"{label}__{_safe(note)}__{meta.id[:8]}.jpg"))
        rows.append((label, note, meta.filename))
        _unlink(thumb)

    # ── summary ──
    n_post = sum(1 for r in rows if r[0] == "POSTABLE")
    base = parse_model_effort(cfg.judge_model)[0]
    total, _bd = compute_cost({base: agg})
    lines = [f"# classify {ts}", "",
             f"model: {cfg.judge_model}   images: {len(rows)}   "
             f"postable: {n_post}   rejected: {len(rows) - n_post}",
             f"cost: {format_total_cost_line(total)}   "
             f"tokens: in={agg['input']} out={agg['output']}", ""]
    for label, note, fn in sorted(rows):
        lines.append(f"- **{label}** — {note}  ({fn})")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== classification summary ===")
    for label, note, fn in sorted(rows):
        print(f"  {label:14} {note[:55]:55}  {fn}")
    print(f"\n{n_post}/{len(rows)} postable   |   {format_total_cost_line(total)} "
          f"(in={agg['input']} out={agg['output']} tok)")
    ui.ok(f"images + summary.md → {out}")
