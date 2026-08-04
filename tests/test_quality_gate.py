"""
Offline tests for the posting quality bar (QUALITY_MIN_SCORE, 2026-07-17).

The judge scores every chosen shot 0-10; run_one_cycle posts only when
safe AND interesting AND quality >= cfg.quality_min_score. These tests prove:
walk-back past a below-bar burst, fail-closed on a missing score, judge_burst's
score validation/clamping, the config default + env override, and that the
prompts actually carry the selective rubric.

They also cover the neighbouring rule that a judge ERROR is not a verdict: a
failed judge call fails closed (never posts) but must leave the burst unseen for
a retry instead of burning it — observed 2026-07-17, two bursts permanently lost
to a transient error:no_client.

Run: .venv/bin/python tests/test_quality_gate.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ic2x.bot as bot  # noqa: E402
import ic2x.judge_burst as jb  # noqa: E402
from ic2x.db import DB  # noqa: E402
from ic2x.status import Status  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ic2x_quality_test_"))
_FIX = _TMP / "fix"; _FIX.mkdir(parents=True, exist_ok=True)


def _img(path: Path, scene: int) -> None:
    im = Image.new("RGB", (256, 256), "black")
    d = ImageDraw.Draw(im)
    if scene == 0:
        d.rectangle([0, 0, 128, 256], fill="white")
    elif scene == 1:
        d.ellipse([48, 48, 208, 208], fill="white")
    else:
        d.rectangle([0, 0, 256, 96], fill="white")
    im.save(path, "JPEG", quality=92)


class _FakeAsset:
    def __init__(self, aid, path): self.id = aid; self._path = path


class FakeIC:
    def __init__(self, assets):  # assets: [(id, fixture_path)] newest-first
        self._assets = assets

    def iter_image_assets(self):
        for aid, path in self._assets:
            yield SimpleNamespace(id=aid), _FakeAsset(aid, path)

    def download(self, asset, version, dest):
        shutil.copy(asset._path, dest); return dest

    def screenshot_ids(self):
        return set()


def _cfg(tag: str, min_q: int = 7, daily_ai_calls: int = 200):
    root = _TMP / tag
    c = SimpleNamespace(
        burst_max_size=5, burst_hamming_threshold=8, burst_max_attempts=3,
        daily_ai_calls=daily_ai_calls, daily_support_calls=1500,
        scene_group_enabled=False, scene_dedup_model="stub",
        hamming_threshold=12, rotation_enabled=False,
        x_dry_run=True, post_max_attempts=3, max_posts_per_day=6, thumb_version="thumb",
        prefetch_concurrency=4, scene_dedup_enabled=False, quality_min_score=min_q,
        keep_reviewed=False, reviewed_dir=root / "reviewed",
        work_dir=root / "work", queue_dir=root / "queue",
        approved_dir=root / "approved", posted_dir=root / "posted", logs_dir=root / "logs",
        scene_thumbs_dir=root / "scene_thumbs", scene_dedup_image_max_px=384,
        judge_model="stub", caption_pass_enabled=False, location_enabled=False,
    )
    for d in (c.work_dir, c.queue_dir, c.approved_dir, c.posted_dir, c.logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return c


def _run_cycle(cfg, db, ic, fake_judge):
    orig_judge, orig_ss = bot.judge_burst, bot.is_screenshot
    bot.judge_burst = fake_judge
    bot.is_screenshot = lambda p: (False, "")  # fixtures carry no EXIF
    try:
        return bot.run_one_cycle(db, cfg, ic, (None, None))
    finally:
        bot.judge_burst, bot.is_screenshot = orig_judge, orig_ss


def test_below_bar_walks_back_and_above_bar_posts():
    """interesting=true but quality below the bar is SKIPPED (marked seen); the
    older above-bar burst posts — the exact 'bar too low' fix."""
    cfg = _cfg("walkback")
    db = DB(_TMP / "walkback.db")
    src = {}
    for aid, scene in [("mediocre", 0), ("great", 1)]:
        p = _FIX / f"{aid}.jpg"; _img(p, scene); src[aid] = p
    ic = FakeIC([("mediocre", src["mediocre"]), ("great", src["great"])])

    calls = {"n": 0}

    def fake_judge(thumbs, cfg_, model_string=None):
        calls["n"] += 1
        q = 5 if calls["n"] == 1 else 9  # both "interesting" — only q9 clears the bar
        return ({"best_index": 0, "safe": True, "interesting": True, "quality": q,
                 "flags": [], "caption": "cap", "reason": f"q{q}"}, 0.1, True)

    outcome = _run_cycle(cfg, db, ic, fake_judge)
    assert outcome == "posted", outcome
    great = db.get_image_by_sha(
        db._conn.execute("SELECT sha256 FROM images WHERE asset_id='great'").fetchone()["sha256"]
    )
    assert great["status"] == Status.POSTED.value
    assert db.seen_asset_id("mediocre") and db.seen_asset_id("great")
    assert db._conn.execute(
        "SELECT COUNT(*) c FROM images WHERE asset_id='mediocre'").fetchone()["c"] == 0
    assert calls["n"] == 2  # mediocre judged + skipped, then great posted
    db.close()


def test_missing_quality_fails_closed():
    """A verdict with no quality key (stub/old model) must never post."""
    cfg = _cfg("noquality")
    db = DB(_TMP / "noquality.db")
    p = _FIX / "nq.jpg"; _img(p, 0)
    ic = FakeIC([("nq", p)])

    def fake_judge(thumbs, cfg_, model_string=None):
        return ({"best_index": 0, "safe": True, "interesting": True,
                 "flags": [], "caption": "cap", "reason": "no score"}, 0.1, True)

    outcome = _run_cycle(cfg, db, ic, fake_judge)
    assert outcome == "exhausted", outcome
    assert db.seen_asset_id("nq")
    assert db.count_posts_rolling_24h() == 0
    db.close()


def test_grouping_calls_do_not_starve_the_judging_budget():
    """Scene grouping fires at every pHash boundary and bills to its OWN meter, so
    the full DAILY_AI_CALLS budget stays available for judging. With both on one
    meter the walk-back stopped after 19 of 195 calls reached the judge (2026-07-17)
    — here a 2-call decision budget must still buy 2 judgments, not 1."""
    cfg = _cfg("budget", daily_ai_calls=2)
    cfg.scene_group_enabled = True
    db = DB(_TMP / "budget.db")
    assets = []
    for i in range(3):                       # three DISTINCT scenes → a boundary each
        p = _FIX / f"bud{i}.jpg"; _img(p, i); assets.append((f"bud{i}", p))
    ic = FakeIC(assets)

    judged = {"n": 0}

    def rejecting(thumbs, cfg_, model_string=None):
        judged["n"] += 1
        return ({"best_index": 0, "safe": True, "interesting": True, "quality": 3,
                 "flags": [], "caption": "c", "reason": "below bar"}, 0.1, True)

    from ic2x import judge_scene_dedup
    orig = judge_scene_dedup.call_scene_dedup
    # "not a duplicate, network was used" → a real grouping call, charged to support
    judge_scene_dedup.call_scene_dedup = lambda *a, **k: (None, True)
    try:
        outcome = _run_cycle(cfg, db, ic, rejecting)
    finally:
        judge_scene_dedup.call_scene_dedup = orig

    stats = db.get_today_stats()
    assert outcome == "ai_cap", outcome
    assert judged["n"] == 2, \
        f"grouping ate the judging budget: only {judged['n']} of 2 bursts judged"
    assert stats["ai_calls"] == 2, stats
    assert stats["support_calls"] > 0, "grouping calls were not tracked at all"
    db.close()


def _err_verdict(flag: str) -> dict:
    """Shape of judge_burst's fail-closed value (see judge_burst._fail)."""
    return {"best_index": 0, "safe": False, "interesting": False, "quality": 0,
            "flags": [flag], "caption": "", "reason": ""}


def test_judge_error_leaves_burst_for_the_next_cycle():
    """A transient judge failure must NOT burn the photos: the burst stays unseen
    and posts on the next cycle (the stream is one-pass, so the retry lands there —
    same contract as a transient download failure). Pre-fix the error committed a
    rejection and the burst was gone from the library for good."""
    cfg = _cfg("judgeerr")
    db = DB(_TMP / "judgeerr.db")
    p = _FIX / "je.jpg"; _img(p, 1)
    ic = FakeIC([("je", p)])

    def erroring(thumbs, cfg_, model_string=None):
        return (_err_verdict("error:no_client"), 0.1, False)

    def good(thumbs, cfg_, model_string=None):
        return ({"best_index": 0, "safe": True, "interesting": True, "quality": 9,
                 "flags": [], "caption": "cap", "reason": "q9"}, 0.1, True)

    assert _run_cycle(cfg, db, ic, erroring) == "exhausted"
    assert not db.seen_asset_id("je"), "a judge error burned the photo — it can never return"
    assert db._conn.execute(
        "SELECT COUNT(*) c FROM images WHERE asset_id='je'").fetchone()["c"] == 0

    assert _run_cycle(cfg, db, ic, good) == "posted", "the retry never got to judge it"
    assert db.get_image_by_sha(
        db._conn.execute("SELECT sha256 FROM images WHERE asset_id='je'").fetchone()["sha256"]
    )["status"] == Status.POSTED.value
    db.close()


def test_judge_error_gives_up_after_max_attempts():
    """The attempts breaker still bounds it — a judge that never recovers marks the
    burst seen on the cfg.burst_max_attempts'th cycle instead of retrying forever."""
    cfg = _cfg("judgeerrloop")
    db = DB(_TMP / "judgeerrloop.db")
    p = _FIX / "jel.jpg"; _img(p, 0)
    ic = FakeIC([("jel", p)])

    def erroring(thumbs, cfg_, model_string=None):
        return (_err_verdict("error:no_client"), 0.1, False)

    for i in range(1, cfg.burst_max_attempts):
        assert _run_cycle(cfg, db, ic, erroring) == "exhausted"
        assert not db.seen_asset_id("jel"), f"gave up too early, on attempt {i}"
    assert _run_cycle(cfg, db, ic, erroring) == "exhausted"
    assert db.seen_asset_id("jel"), "a permanently failing judge must stop retrying"
    db.close()


def test_model_refusal_is_a_verdict_not_an_error():
    """`model_refused` is the model declining on content — a real rejection. It must
    still commit seen on the first call, not get retried as a blip."""
    cfg = _cfg("refused")
    db = DB(_TMP / "refused.db")
    p = _FIX / "rf.jpg"; _img(p, 1)
    ic = FakeIC([("rf", p)])

    calls = {"n": 0}

    def fake_judge(thumbs, cfg_, model_string=None):
        calls["n"] += 1
        return (_err_verdict("model_refused"), 0.1, True)

    outcome = _run_cycle(cfg, db, ic, fake_judge)
    assert outcome == "exhausted", outcome
    assert db.seen_asset_id("rf") and calls["n"] == 1, \
        f"a refusal must reject immediately, got {calls['n']} calls"
    db.close()


def test_judge_burst_validates_quality():
    """judge_burst clamps quality to an int 0-10; missing/garbage → 0."""
    fixture = _FIX / "jv.jpg"; _img(fixture, 1)
    cfg = SimpleNamespace(judge_model="qwen3.5-flash", ollama_base_url="",
                          judge_image_max_px=512, judge_extra_rules="")

    cases = [({}, 0), ({"quality": "8"}, 8), ({"quality": 15}, 10),
             ({"quality": -3}, 0), ({"quality": "n/a"}, 0), ({"quality": 7.6}, 7)]
    orig = jb.call_vision_judge_multi
    try:
        for raw_quality, expected in cases:
            def fake_call(*, model_string, ollama_base_url, call, usage_out=None,
                          _rq=raw_quality):
                return ({"best_index": 0, "safe": True, "interesting": True,
                         "caption": "", **_rq}, 0.1, True, True)
            jb.call_vision_judge_multi = fake_call
            v, _el, _net = jb.judge_burst([fixture], cfg)
            assert v["quality"] == expected, (raw_quality, v["quality"])
    finally:
        jb.call_vision_judge_multi = orig


def test_config_default_and_env_override():
    from ic2x.config import load_config
    old = os.environ.pop("QUALITY_MIN_SCORE", None)
    try:
        os.environ["QUALITY_MIN_SCORE"] = "9"
        assert load_config().quality_min_score == 9
        del os.environ["QUALITY_MIN_SCORE"]
        assert load_config().quality_min_score == 7  # code default
    finally:
        if old is not None:
            os.environ["QUALITY_MIN_SCORE"] = old


def test_prompts_carry_selective_rubric():
    """The shared rubric and both judge schemas must keep the selective bar."""
    from ic2x.judge_safety_quality import JUDGE_PROMPT, QUALITY_BLOCK
    for needle in ('"quality"', "0-10", "Automatic caps", "far away",
                   "do NOT default to 7", "When unsure, set interesting=false"):
        assert needle in QUALITY_BLOCK, needle
    assert '"quality"' in JUDGE_PROMPT
    cfg = SimpleNamespace(judge_extra_rules="")
    assert '"quality"' in jb.burst_prompt(cfg)
    assert "Automatic caps" in jb.burst_prompt(cfg)


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
