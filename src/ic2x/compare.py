"""
`ic2x compare` — run the same recent bursts through two judge models, side by
side, with NO posting and NO mutation of the real seen-set (a throwaway DB).
Prints each model's best_index / decisions plus token cost, to pick the default.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ic2x.bot import ICloudAssetSource, find_next_burst, _unlink
from ic2x.config import ensure_dirs, load_config
from ic2x.db import DB
from ic2x.icloud_photos import ICloudPhotos, ReauthRequired
from ic2x.judge_burst import judge_burst
from ic2x.utils import ui
from ic2x.utils.ai_client import get_run_usage, reset_run_usage
from ic2x.utils.cost_report import compute_cost, format_total_cost_line

logger = logging.getLogger("ic2x.compare")


def compare(models: list[str], n_bursts: int = 5) -> None:
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

    # Throwaway DB: assemble real bursts without touching the bot's seen-set.
    tmp_path = cfg.work_dir / "compare_tmp.db"
    _unlink(tmp_path)
    db = DB(tmp_path)

    ui.info(f"Indexing recent photos for {n_bursts} bursts …")
    limit = n_bursts * cfg.burst_max_size * 4 + 20
    indexed = 0
    for meta, _ in ic.iter_image_assets():
        db.upsert_asset(meta.id, meta.created, meta.filename,
                        is_live=meta.is_live, width=meta.width, height=meta.height)
        indexed += 1
        if indexed >= limit:
            break
    ss = ic.screenshot_ids()
    if ss:
        db.mark_screenshots(list(ss))

    source = ICloudAssetSource(ic, cfg)
    bursts = []
    while len(bursts) < n_bursts:
        b = find_next_burst(db, source, cfg, set())
        if b is None:
            break
        if b.members:
            bursts.append(b)
        db.commit_burst([m.asset_id for m in b.members] + b.aux_seen, None)

    if not bursts:
        ui.warn("No bursts assembled (all screenshots / library empty?).")
        db.close()
        return
    ui.ok(f"Assembled {len(bursts)} burst(s): sizes {[len(b.members) for b in bursts]}")

    # Run each model over every burst; isolate token usage per model.
    results: dict[str, dict] = {}
    for model in models:
        reset_run_usage()
        verdicts = []
        for b in bursts:
            verdict, _el, _net = judge_burst([m.thumb for m in b.members], cfg, model_string=model)
            verdicts.append(verdict)
        total, _bd = compute_cost(get_run_usage())
        results[model] = {"verdicts": verdicts, "cost": total, "usage": get_run_usage()}

    _print_report(models, bursts, results)
    _write_artifact(cfg, models, bursts, results)

    # cleanup thumbnails + throwaway db
    for b in bursts:
        for m in b.members:
            _unlink(m.thumb)
    db.close()
    _unlink(tmp_path)


def _print_report(models, bursts, results) -> None:
    print("\n=== model comparison (no posting) ===")
    for bi, b in enumerate(bursts):
        print(f"\nBurst {bi}  (n={len(b.members)})")
        for model in models:
            v = results[model]["verdicts"][bi]
            print(f"  {model:28} best={v.get('best_index')}  safe={v.get('safe')}  "
                  f"interesting={v.get('interesting')}  caption={v.get('caption','')!r}")
    print("\n--- cost ---")
    for model in models:
        usage = results[model]["usage"]
        toks = {m: (u["input"], u["output"]) for m, u in usage.items()}
        print(f"  {model:28} {format_total_cost_line(results[model]['cost'])}   tokens={toks}")
    # best_index agreement
    agree = sum(
        1 for bi in range(len(bursts))
        if len({results[m]['verdicts'][bi].get('best_index') for m in models}) == 1
    )
    print(f"\nbest_index agreement: {agree}/{len(bursts)} bursts")


def _write_artifact(cfg, models, bursts, results) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = cfg.logs_dir / f"compare_{ts}.md"
    try:
        lines = [f"# model comparison {ts}", "", f"bursts: {len(bursts)}  models: {', '.join(models)}", ""]
        for bi, b in enumerate(bursts):
            lines.append(f"## burst {bi} (n={len(b.members)})")
            for model in models:
                v = results[model]["verdicts"][bi]
                lines.append(f"- **{model}**: best={v.get('best_index')} safe={v.get('safe')} "
                             f"interesting={v.get('interesting')} caption={v.get('caption','')!r}")
            lines.append("")
        lines.append("## cost")
        for model in models:
            lines.append(f"- **{model}**: {format_total_cost_line(results[model]['cost'])}")
        out.write_text("\n".join(lines), encoding="utf-8")
        ui.info(f"wrote {out}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write compare artifact: %s", exc)
