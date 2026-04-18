"""
Terminal review UI for `ic2x review`.

For each image in queue/ (oldest first):
  [y] post     → move to approved/, mark DB approved
  [n] skip     → move to rejected/quality/, mark DB rejected
  [e] edit     → open $EDITOR with caption, then treat as y
  [q] quit     → clean exit, nothing left ambiguous
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ic2x.config import load_config, ensure_dirs
from ic2x.db import DB
from ic2x.utils import ui


def review() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    db = DB(cfg.db_path)

    queue_items = _load_queue(cfg.queue_dir)
    if not queue_items:
        ui.info("Queue is empty — nothing to review.")
        db.close()
        return

    ui.review_section_banner(len(queue_items))

    for idx, (jpg_path, meta) in enumerate(queue_items, start=1):
        filename   = jpg_path.name
        phash      = jpg_path.stem
        description = meta.get("description", "")
        caption    = meta.get("caption", "")
        reason     = meta.get("reason", "")

        ui.review_header(idx, len(queue_items), filename, description, caption, reason)

        # Open image in system viewer (non-blocking)
        _open_image(jpg_path)

        while True:
            ui.review_prompt()
            try:
                choice = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                ui.info("Interrupted — exiting review.")
                db.close()
                return

            if choice == "y":
                _approve(jpg_path, phash, caption, cfg, db)
                ui.ok(f"{filename} → approved")
                break

            elif choice == "n":
                _skip(jpg_path, phash, cfg, db)
                ui.info(f"{filename} → skipped")
                break

            elif choice == "e":
                edited = _edit_caption(caption)
                if edited is not None:
                    caption = edited
                    ui.info(f"Caption updated: {caption}")
                    _approve(jpg_path, phash, caption, cfg, db)
                    ui.ok(f"{filename} → approved (edited caption)")
                else:
                    ui.warn("Editor returned empty caption — skipping approve. Press y/n/q.")
                    continue
                break

            elif choice == "q":
                ui.info("Quit — exiting review.")
                db.close()
                return

            else:
                ui.warn("Unknown key. Use y / n / e / q.")

    ui.info("Review complete.")
    db.close()


def _load_queue(queue_dir: Path) -> list[tuple[Path, dict]]:
    """Return (jpg_path, metadata_dict) sorted oldest first."""
    items = []
    for jpg in sorted(queue_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime):
        json_path = jpg.with_suffix(".json")
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        else:
            meta = {}
        items.append((jpg, meta))
    return items


def _approve(jpg_path: Path, phash: str, caption: str, cfg, db: DB) -> None:
    dest = cfg.approved_dir / jpg_path.name
    shutil.move(str(jpg_path), dest)
    # Move sidecar JSON too
    json_src = jpg_path.with_suffix(".json")
    if json_src.exists():
        shutil.move(str(json_src), cfg.approved_dir / json_src.name)
    # Look up sha256 from DB via phash
    sha = _sha_for_phash(db, phash)
    if sha:
        db.set_status(sha, "approved", caption=caption)


def _skip(jpg_path: Path, phash: str, cfg, db: DB) -> None:
    dest_dir = cfg.rejected_dir / "quality"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(jpg_path), dest_dir / jpg_path.name)
    json_src = jpg_path.with_suffix(".json")
    if json_src.exists():
        json_src.unlink()
    sha = _sha_for_phash(db, phash)
    if sha:
        db.set_status(sha, "rejected", reject_stage="quality", reject_reason="reviewer_skip")


def _sha_for_phash(db: DB, phash: str) -> str | None:
    row = db._conn.execute(
        "SELECT sha256 FROM images WHERE phash = ? ORDER BY id DESC LIMIT 1", (phash,)
    ).fetchone()
    return row["sha256"] if row else None


def _open_image(path: Path) -> None:
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _edit_caption(current: str) -> str | None:
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(current)
        tmp = f.name
    try:
        subprocess.run([editor, tmp], check=True)
        result = Path(tmp).read_text(encoding="utf-8").strip()
        return result if result else None
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
