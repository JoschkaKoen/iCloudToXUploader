"""
Offline tests for the same-scene dedup gate + module. No network, no iCloud.

Proves: (1) run_one_cycle skips a winner the scene-dedup flags as a duplicate of a
recent post and walks back to a different scene; (2) when scene-dedup says "not a
duplicate", the post proceeds; (3) call_scene_dedup fails OPEN on bad output and
returns the right 0-based index on a valid match.

Run: .venv/bin/python tests/test_scene_dedup.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ic2x.bot as bot  # noqa: E402
from ic2x import judge_scene_dedup  # noqa: E402
from ic2x.db import DB  # noqa: E402
from ic2x.status import Status  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_scene_test_"))
_FIX = _TMP / "fix"; _FIX.mkdir(parents=True, exist_ok=True)
_SEQ = [0]


def _img(path: Path, scene: int) -> None:
    im = Image.new("RGB", (256, 256), "black"); d = ImageDraw.Draw(im)
    if scene == 0:
        d.rectangle([0, 0, 128, 256], fill="white")
    elif scene == 1:
        d.ellipse([48, 48, 208, 208], fill="white")
    else:
        d.rectangle([0, 0, 256, 96], fill="white")  # scene 2+: distinct top strip
    im.save(path, "JPEG", quality=92)


class _FakeAsset:
    def __init__(self, aid, path): self.id = aid; self._path = path


class FakeIC:
    def __init__(self, assets): self._assets = assets

    def iter_image_assets(self):
        for aid, path in self._assets:
            yield SimpleNamespace(id=aid, filename=f"{aid}.jpg"), _FakeAsset(aid, path)

    def download(self, asset, version, dest): shutil.copy(asset._path, dest); return dest

    def screenshot_ids(self): return set()


def _cfg():
    _SEQ[0] += 1
    base = _TMP / f"run{_SEQ[0]}"
    c = SimpleNamespace(
        burst_max_size=5, burst_hamming_threshold=8, burst_max_attempts=3,
        daily_ai_calls=200, hamming_threshold=12, rotation_enabled=False,
        x_dry_run=True, post_max_attempts=3, max_posts_per_day=6, thumb_version="thumb",
        prefetch_concurrency=4,
        scene_dedup_enabled=True, scene_dedup_model="qwen3-vl-flash",
        scene_dedup_recent_n=6, scene_dedup_image_max_px=384,
        scene_thumbs_dir=base / "scene_thumbs",
        keep_reviewed=False, reviewed_dir=base / "reviewed",
        work_dir=base / "work", queue_dir=base / "queue",
        approved_dir=base / "approved", posted_dir=base / "posted", logs_dir=base / "logs",
        judge_model="stub", ollama_base_url="http://localhost:11434/v1",
        caption_pass_enabled=False, location_enabled=False,
    )
    for d in (c.work_dir, c.queue_dir, c.approved_dir, c.posted_dir, c.logs_dir, c.scene_thumbs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return c


def _seed_recent_post(db, cfg, phash="ffffffffffffffff"):
    """A prior POSTED image + its scene_thumb, so the gate has something to compare."""
    db._conn.execute(
        "INSERT INTO images (asset_id, sha256, phash, status, posted_at) VALUES (?,?,?,?,?)",
        ("seed", "seedsha", phash, Status.POSTED.value, "2026-01-01T00:00:00"),
    )
    db._conn.commit()
    _img(cfg.scene_thumbs_dir / f"{phash}.jpg", 0)


def _postable_judge(thumbs, cfg_, model_string=None):
    return ({"best_index": 0, "safe": True, "interesting": True, "quality": 9,
             "flags": [], "caption": "nice", "reason": "good"}, 0.1, True)


def _run(cfg, db, ic, scene_dedup_fn):
    orig = (bot.judge_burst, bot.is_screenshot, judge_scene_dedup.call_scene_dedup)
    bot.judge_burst = _postable_judge
    bot.is_screenshot = lambda p: (False, "")
    judge_scene_dedup.call_scene_dedup = scene_dedup_fn
    try:
        return bot.run_one_cycle(db, cfg, ic, (None, None))
    finally:
        bot.judge_burst, bot.is_screenshot, judge_scene_dedup.call_scene_dedup = orig


def test_cycle_skips_scene_dup_and_walks_back():
    cfg = _cfg(); db = DB(_TMP / "a.db")
    a = _FIX / "a.jpg"; _img(a, 0); b = _FIX / "b.jpg"; _img(b, 1)  # far pHash → 2 bursts
    ic = FakeIC([("A", a), ("B", b)])
    _seed_recent_post(db, cfg)
    n = {"i": 0}

    def sd(cand, recent, cfg_, model_string=None):
        n["i"] += 1
        return (0, True) if n["i"] == 1 else (None, True)  # A = dup of seed, B = not

    outcome = _run(cfg, db, ic, sd)
    assert outcome == "posted", outcome
    assert db.seen_asset_id("A") and db.seen_asset_id("B")
    # B posted; A was skipped as a scene-dup BEFORE _prepare_winner (no images row)
    brow = db._conn.execute("SELECT sha256 FROM images WHERE asset_id='B'").fetchone()
    assert brow and db.get_image_by_sha(brow["sha256"])["status"] == Status.POSTED.value
    assert db._conn.execute("SELECT 1 FROM images WHERE asset_id='A'").fetchone() is None
    assert n["i"] == 2


def test_cycle_posts_when_not_dup():
    cfg = _cfg(); db = DB(_TMP / "c.db")
    a = _FIX / "c.jpg"; _img(a, 1)
    ic = FakeIC([("A", a)])
    _seed_recent_post(db, cfg)
    outcome = _run(cfg, db, ic, lambda *a, **k: (None, True))  # not a dup → posts
    assert outcome == "posted", outcome
    row = db._conn.execute("SELECT sha256 FROM images WHERE asset_id='A'").fetchone()
    assert db.get_image_by_sha(row["sha256"])["status"] == Status.POSTED.value


def test_grouping_merges_same_scene_variants():
    cfg = _cfg(); cfg.scene_group_enabled = True
    db = DB(_TMP / "group.db")
    port = _FIX / "gp.jpg"; _img(port, 0)
    land = _FIX / "gl.jpg"; _img(land, 1)    # pHash-distinct from port (boundary 1)
    other = _FIX / "go.jpg"; _img(other, 2)  # pHash-distinct from land (boundary 2)
    ic = FakeIC([("port", port), ("land", land), ("other", other)])
    calls = {"n": 0}

    def fake_same(cand, head, cfg_, db_):
        calls["n"] += 1
        return calls["n"] == 1   # boundary 1 (land vs port) → same scene; boundary 2 → different

    orig = bot._ai_same_scene
    bot._ai_same_scene = fake_same
    try:
        ss = set()
        stream = bot._Stream(ic.iter_image_assets(), db, cfg, ic, ss, concurrency=2)
        bursts = []
        while True:
            b = bot.assemble_burst(stream, cfg, ic, ss, db)
            if b is None:
                break
            if b.members:
                bursts.append([m.asset_id for m in b.members])
            db.commit_burst([m.asset_id for m in b.members] + b.aux_seen, None)
        stream.close()
    finally:
        bot._ai_same_scene = orig
    assert bursts == [["port", "land"], ["other"]], bursts   # port+land MERGED; other separate
    assert calls["n"] == 2


def test_module_fails_open_on_bad_output():
    cfg = _cfg()
    cand = _FIX / "x.jpg"; _img(cand, 0); rec = _FIX / "y.jpg"; _img(rec, 0)
    orig = judge_scene_dedup.call_vision_judge_multi
    try:
        judge_scene_dedup.call_vision_judge_multi = lambda **k: ({"duplicate_of": 0}, 0.0, False, True)
        assert judge_scene_dedup.call_scene_dedup(cand, [rec], cfg) == (None, True)   # client error
        judge_scene_dedup.call_vision_judge_multi = lambda **k: ({"duplicate_of": "nope"}, 0.0, True, True)
        assert judge_scene_dedup.call_scene_dedup(cand, [rec], cfg) == (None, True)   # malformed
        judge_scene_dedup.call_vision_judge_multi = lambda **k: ({"duplicate_of": 9}, 0.0, True, True)
        assert judge_scene_dedup.call_scene_dedup(cand, [rec], cfg) == (None, True)   # out of range
        judge_scene_dedup.call_vision_judge_multi = lambda **k: ({"duplicate_of": 1}, 0.0, True, True)
        assert judge_scene_dedup.call_scene_dedup(cand, [rec], cfg) == (0, True)      # valid match → 0-based
    finally:
        judge_scene_dedup.call_vision_judge_multi = orig


def test_module_empty_recent():
    cfg = _cfg(); cand = _FIX / "z.jpg"; _img(cand, 0)
    assert judge_scene_dedup.call_scene_dedup(cand, [], cfg) == (None, False)


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
