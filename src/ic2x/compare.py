"""
`ic2x compare` — run the same recent bursts through two judge models, side by
side, with NO posting and NO mutation of the real seen-set (a throwaway DB).
Prints each model's best_index / decisions plus token cost, to pick the default.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ic2x.bot import _Stream, _unlink, assemble_burst
from ic2x.config import ensure_dirs, load_config
from ic2x.db import DB
from ic2x.icloud_photos import ICloudPhotos, ReauthRequired
from ic2x.judge_burst import judge_burst
from ic2x.utils import ui
from ic2x.utils.ai_client import parse_model_effort
from ic2x.utils.cost_report import compute_cost, format_total_cost_line

logger = logging.getLogger("ic2x.compare")


def compare(models: list[str], n_bursts: int = 5, keep: bool = False) -> None:
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

    ui.info(f"Assembling {n_bursts} recent burst(s) …")
    screenshot_ids = ic.screenshot_ids()
    stream = _Stream(ic.iter_image_assets(), db)
    bursts = []
    while len(bursts) < n_bursts:
        b = assemble_burst(stream, cfg, ic, screenshot_ids)
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

    # Run EVERY (model × burst) call concurrently in one pool. Cost is attributed
    # per call via usage_out, so two variants sharing a base model name (qwen
    # ", off" vs ", 3000") don't intermix — no need to serialize the models.
    from concurrent.futures import ThreadPoolExecutor

    tasks = [(mi, bi) for mi in range(len(models)) for bi in range(len(bursts))]

    def _run(task):
        mi, bi = task
        usage: dict = {}
        v, _el, _net = judge_burst([m.thumb for m in bursts[bi].members], cfg,
                                   model_string=models[mi], usage_out=usage)
        return mi, bi, v, usage

    with ThreadPoolExecutor(max_workers=min(16, len(tasks))) as ex:
        outs = list(ex.map(_run, tasks))

    results: dict[str, dict] = {}
    for mi, model in enumerate(models):
        verdicts: list = [None] * len(bursts)
        agg = {"input": 0, "output": 0, "thinking": 0}
        for omi, bi, v, usage in outs:
            if omi == mi:
                verdicts[bi] = v
                agg["input"] += usage.get("input", 0)
                agg["output"] += usage.get("output", 0)
        base = parse_model_effort(model)[0]
        total, _bd = compute_cost({base: agg})
        results[model] = {"verdicts": verdicts, "cost": total, "usage": {base: agg}}

    _print_report(models, bursts, results)
    _write_artifact(cfg, models, bursts, results)
    if keep:
        _keep_images(cfg, models, bursts, results)

    # cleanup thumbnails + throwaway db
    for b in bursts:
        for m in b.members:
            _unlink(m.thumb)
    db.close()
    _unlink(tmp_path)


def _keep_images(cfg, models, bursts, results) -> None:
    """Save each evaluated thumbnail + a verdicts.md so decisions are reviewable."""
    import shutil
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = cfg.logs_dir.parent / "compare_out" / ts
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# compare {ts}", "", f"models: {', '.join(models)}", ""]
    n_saved = 0
    for bi, b in enumerate(bursts):
        names = []
        for j, m in enumerate(b.members):
            dst = out / f"burst{bi}_img{j}.jpg"
            try:
                shutil.copy(m.thumb, dst); names.append(dst.name); n_saved += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("keep: could not copy %s: %s", m.thumb, exc)
        lines.append(f"## burst {bi}  ({', '.join(names)})")
        for model in models:
            v = results[model]["verdicts"][bi]
            lines.append(f"- **{model}**: best={v.get('best_index')} safe={v.get('safe')} "
                         f"interesting={v.get('interesting')} caption={v.get('caption','')!r}")
        lines.append("")
    (out / "verdicts.md").write_text("\n".join(lines), encoding="utf-8")
    ui.ok(f"saved {n_saved} thumbnail(s) + verdicts.md to {out}")


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
