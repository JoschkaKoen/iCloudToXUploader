"""
`ic2x scene-dedup-test --dir <folder>` — calibrate the same-scene dedup offline.
No posting, no DB, no iCloud. The lexicographically-FIRST image in <folder> is the
candidate; the rest are treated as "recent posts". Prints whether the candidate is
judged the SAME scene as any of them (which one + reason + cost) and copies the
compared images to scene_dedup_out/<ts>/ for review.

Drop in pairs to calibrate: a same-scene portrait+landscape pair should MATCH;
two different dishes / sunsets should NOT.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ic2x.config import ensure_dirs, load_config
from ic2x.judge_scene_dedup import call_scene_dedup
from ic2x.utils import ui
from ic2x.utils.ai_client import get_run_usage, reset_run_usage
from ic2x.utils.cost_report import compute_cost, format_total_cost_line

logger = logging.getLogger("ic2x.scene_dedup_test")

_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def scene_dedup_test(folder: str, model: str | None = None) -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    from ic2x.utils.logging_setup import setup_logging
    setup_logging(cfg.logs_dir)

    src = Path(folder).expanduser()
    if not src.is_dir():
        ui.err(f"not a folder: {src}")
        return
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in _EXTS)
    if len(imgs) < 2:
        ui.err(f"need >=2 images in {src} (first = candidate, rest = 'recent posts'); found {len(imgs)}")
        return

    candidate, recent = imgs[0], imgs[1:]
    model_str = model or cfg.scene_dedup_model
    ui.info(f"candidate: {candidate.name}  vs {len(recent)} other(s)  [model {model_str}]")

    reset_run_usage()
    dup, _used = call_scene_dedup(candidate, recent, cfg, model_string=model_str)
    total, _bd = compute_cost(get_run_usage())

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = cfg.logs_dir.parent / "scene_dedup_out" / ts
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(candidate, out / f"00_CANDIDATE__{candidate.name}")
    for i, p in enumerate(recent, 1):
        tag = "MATCH" if dup == i - 1 else "other"
        shutil.copy(p, out / f"{i:02d}_{tag}__{p.name}")

    print("\n=== scene-dedup-test ===")
    if dup is None:
        print(f"  NO MATCH — '{candidate.name}' is a different scene from all {len(recent)} other(s)")
    else:
        print(f"  MATCH — '{candidate.name}' is the SAME scene as '{recent[dup].name}' (#{dup + 1})")
    print(f"  {format_total_cost_line(total)}")
    ui.ok(f"compared images → {out}")
