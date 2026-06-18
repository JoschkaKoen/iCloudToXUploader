"""
Offline integration test for run_one_cycle — the orchestration heart.

Stubs the VLM judge and the EXIF screenshot gate; everything else (burst
assembly, dedup, prepare, atomic commit, dry-run post, seen-set advance) runs for
real. Proves walk-back-until-postable: a boring newest burst is rejected and the
bot falls through to an older interesting one and posts it.

Run: .venv/bin/python tests/test_cycle.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ic2x.bot as bot  # noqa: E402
from ic2x.db import DB  # noqa: E402
from ic2x.status import Status  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_cycle_test_"))
_FIX = _TMP / "fix"; _FIX.mkdir(parents=True, exist_ok=True)


def _img(path: Path, scene: int) -> None:
    im = Image.new("RGB", (256, 256), "black")
    d = ImageDraw.Draw(im)
    if scene == 0:
        d.rectangle([0, 0, 128, 256], fill="white")
    else:
        d.ellipse([48, 48, 208, 208], fill="white")
    im.save(path, "JPEG", quality=92)


class FakeSource:
    def __init__(self, mapping):
        self._map = mapping; self._n = 0

    def download_thumb(self, asset_id):
        self._n += 1
        dst = _TMP / f"t_{self._n}.jpg"; shutil.copy(self._map[asset_id], dst); return dst

    def download_original(self, asset_id):
        self._n += 1
        dst = _TMP / f"o_{self._n}.jpg"; shutil.copy(self._map[asset_id], dst); return dst


def _cfg():
    c = SimpleNamespace(
        burst_max_size=5, burst_hamming_threshold=8, burst_max_attempts=3,
        daily_ai_calls=200, hamming_threshold=12, rotation_enabled=False,
        x_dry_run=True, post_max_attempts=3, max_posts_per_day=6,
        work_dir=_TMP / "work", queue_dir=_TMP / "queue",
        approved_dir=_TMP / "approved", posted_dir=_TMP / "posted", logs_dir=_TMP / "logs",
        judge_model="stub",
    )
    for d in (c.work_dir, c.queue_dir, c.approved_dir, c.posted_dir, c.logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return c


def test_walk_back_until_postable_then_post():
    cfg = _cfg()
    db = DB(_TMP / "cycle.db")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # newest = "boring" (scene 0); older = "good" (scene 1, far pHash so a separate burst)
    src_map = {}
    for aid, scene, when in [("boring", 0, base + timedelta(hours=2)),
                             ("good", 1, base + timedelta(hours=1))]:
        p = _FIX / f"{aid}.jpg"; _img(p, scene); src_map[aid] = p
        db.upsert_asset(aid, when, f"{aid}.HEIC")

    # judge: first burst (boring) → not interesting; second (good) → interesting, best 0
    calls = {"n": 0}

    def fake_judge(thumbs, cfg_, model_string=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return ({"best_index": 0, "safe": True, "interesting": False,
                     "flags": [], "caption": "", "reason": "boring"}, 0.1, True)
        return ({"best_index": 0, "safe": True, "interesting": True,
                 "flags": [], "caption": "nice", "reason": "good"}, 0.1, True)

    orig_judge, orig_ss = bot.judge_burst, bot.is_screenshot
    bot.judge_burst = fake_judge
    bot.is_screenshot = lambda p: (False, "")  # camera photo (fixtures carry no EXIF)
    try:
        outcome = bot.run_one_cycle(db, cfg, FakeSource(src_map), (None, None))
    finally:
        bot.judge_burst, bot.is_screenshot = orig_judge, orig_ss

    assert outcome == "posted", outcome
    # the GOOD one was posted; the BORING one was decided-seen, not posted
    good = [r for r in db._conn.execute("SELECT * FROM images").fetchall()
            if r["asset_id"] == "good"]
    assert good and good[0]["status"] == Status.POSTED.value
    assert db.count_posts_rolling_24h() == 1
    # both assets are now seen → none re-assembled
    assert not db.has_unseen_assets()
    assert calls["n"] == 2  # judged boring, then good


def _main() -> int:
    failed = 0
    for name, t in sorted(globals().items()):
        if name.startswith("test_") and callable(t):
            try:
                t(); print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                import traceback; print(f"FAIL {name}: {exc}"); traceback.print_exc()
    print("OK" if not failed else "FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
