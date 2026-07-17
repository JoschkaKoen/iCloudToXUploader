"""
Offline test for `ic2x bot --test` (fast dry-run soak). No network, no iCloud, no
real posts. Proves --test (1) forces dry-run + keep_reviewed, (2) puts every run in
its OWN test_run/<timestamp>/ folder — leaving a prior run's folder, the real
state.db, and icloud_cookie_dir untouched, (3) cycles fast and stops cleanly at
--test-cycles, dry-posting along the way.

Run: .venv/bin/python tests/test_bot_testmode.py   (or via pytest)
"""

from __future__ import annotations

import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# This test builds a real Config via load_config(), which requires credential env
# vars. No real network calls are made (clients are monkeypatched below), so dummy
# values suffice. setdefault means a real .env / environment always wins. Mirrors
# tests/conftest.py so this file also runs standalone (pytest-free).
for _k, _v in {
    "ICLOUD_USERNAME": "test@example.com", "ICLOUD_PASSWORD": "test-password",
    "TWITTER_CONSUMER_KEY": "test-ck", "TWITTER_CONSUMER_SECRET": "test-cs",
    "TWITTER_ACCESS_TOKEN": "test-at", "TWITTER_ACCESS_TOKEN_SECRET": "test-ats",
}.items():
    os.environ.setdefault(_k, _v)

import ic2x.bot as bot  # noqa: E402
from ic2x.config import _PROJECT_ROOT  # noqa: E402
from ic2x.db import DB  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_testmode_"))
_FIX = _TMP / "fix"
_FIX.mkdir(parents=True, exist_ok=True)


def _img(path: Path, scene: int) -> None:
    im = Image.new("RGB", (256, 256), "black")
    d = ImageDraw.Draw(im)
    if scene == 0:
        d.rectangle([0, 0, 128, 256], fill="white")
    elif scene == 1:
        d.ellipse([40, 40, 216, 216], fill="white")
    else:
        d.rectangle([0, 0, 256, 96], fill="white")
    im.save(path, "JPEG", quality=92)


class _FakeAsset:
    def __init__(self, aid, path):
        self.id = aid
        self._path = path


class FakeIC:
    def __init__(self, assets):
        self._assets = assets

    def ensure_session(self):
        pass

    def iter_image_assets(self):
        for aid, path in self._assets:
            yield SimpleNamespace(id=aid, filename=f"{aid}.jpg"), _FakeAsset(aid, path)

    def download(self, asset, version, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)  # real download does this too
        shutil.copy(asset._path, dest)
        return dest

    def screenshot_ids(self):
        return set()


def _postable_judge(thumbs, cfg_, model_string=None):
    return ({"best_index": 0, "safe": True, "interesting": True, "flags": [],
             "caption": "a nice test scene 🌅", "reason": "good"}, 0.1, True)


def test_bot_test_mode_isolates_state_and_bounds_cycles():
    # distinct-scene images → separate bursts → multiple dry-posts available
    imgs = [(f"A{i}", _FIX / f"s{i}.jpg") for i in range(4)]
    for i, (_, p) in enumerate(imgs):
        _img(p, i % 3)

    real_db = _PROJECT_ROOT / "state.db"
    real_db_existed = real_db.exists()
    real_db_mtime = real_db.stat().st_mtime if real_db_existed else None

    # simulate an earlier run's folder — it must survive this run untouched
    test_root = _PROJECT_ROOT / "test_run"
    prior = test_root / "0000-00-00_000000"
    prior.mkdir(parents=True, exist_ok=True)
    (prior / "marker.txt").write_text("keep me")

    # real cfg, but with the networked AI features off so the soak runs offline
    cfg = bot.load_config()
    cfg.x_dry_run = True                 # dry speed-run → isolated state (regardless of env)
    cfg.scene_dedup_enabled = False
    cfg.scene_group_enabled = False
    cfg.rotation_enabled = False
    cfg.location_enabled = False
    cfg.owner_check_enabled = False     # keep the soak offline (no owner-gate VLM call)
    cfg.local_prefilter_enabled = False  # keep the soak hermetic (no local Ollama calls)
    cfg.local_judge_shadow = False
    cfg.color_enhance_enabled = False   # keep the soak offline (no Aliyun calls)
    cfg.caption_pass_enabled = False    # keep the soak offline (no caption VLM call)
    cfg.reconcile_on_startup = False    # keep the soak offline (no X reads)
    real_cookie = cfg.icloud_cookie_dir

    bot._stop = False
    sigint, sigterm = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    orig = (bot.load_config, bot.require_vision_api_credentials, bot.ICloudPhotos,
            bot.make_clients, bot.judge_burst, bot.is_screenshot)
    bot.load_config = lambda: cfg
    bot.require_vision_api_credentials = lambda *a, **k: None
    bot.ICloudPhotos = lambda c: FakeIC(imgs)
    bot.make_clients = lambda c: (None, None)
    bot.judge_burst = _postable_judge
    bot.is_screenshot = lambda p: (False, "")
    try:
        bot.bot(test=True, test_cycles=2)
    finally:
        (bot.load_config, bot.require_vision_api_credentials, bot.ICloudPhotos,
         bot.make_clients, bot.judge_burst, bot.is_screenshot) = orig
        signal.signal(signal.SIGINT, sigint)
        signal.signal(signal.SIGTERM, sigterm)

    run_root = cfg.db_path.parent
    try:
        # forced dry + keep_reviewed
        assert cfg.x_dry_run is True and cfg.keep_reviewed is True
        # this run lives in its OWN test_run/<timestamp>/ folder …
        assert run_root.parent == test_root, f"run folder not under test_run/: {run_root}"
        assert run_root != prior and (prior / "marker.txt").exists(), \
            "a prior run's folder was wiped — runs must not overwrite each other!"
        # … with ALL state redirected under it
        assert cfg.db_path == run_root / "state.db" and cfg.db_path.exists()
        assert cfg.work_dir == run_root / "work"
        assert cfg.queue_dir == run_root / "queue"
        assert cfg.approved_dir == run_root / "approved"
        assert cfg.posted_dir == run_root / "posted"
        assert cfg.scene_thumbs_dir == run_root / "scene_thumbs"
        assert cfg.reviewed_dir == run_root / "reviewed"
        assert cfg.logs_dir == run_root / "logs"
        # … except the 2FA cookie dir, which must stay real
        assert cfg.icloud_cookie_dir == real_cookie
        # the real state.db is untouched (not created/modified by the soak)
        if real_db_existed:
            assert real_db.stat().st_mtime == real_db_mtime, "real state.db was modified!"
        else:
            assert not real_db.exists(), "real state.db was created by a test run!"
        # it actually cycled + dry-posted into the isolated db
        tdb = DB(run_root / "state.db")
        posted = tdb._conn.execute(
            "SELECT COUNT(*) c FROM images WHERE status='posted'").fetchone()["c"]
        tdb.close()
        assert posted >= 1, f"expected >=1 dry-post, got {posted}"
    finally:
        shutil.rmtree(run_root, ignore_errors=True)
        shutil.rmtree(prior, ignore_errors=True)
        try:
            test_root.rmdir()      # remove only if now empty (don't disturb real runs)
        except OSError:
            pass


def _main() -> int:
    try:
        test_bot_test_mode_isolates_state_and_bounds_cycles()
        print("PASS test_bot_test_mode_isolates_state_and_bounds_cycles")
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"FAIL: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
