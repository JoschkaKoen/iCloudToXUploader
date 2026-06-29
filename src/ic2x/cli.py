"""
ic2x — autonomous iCloud → X "best-of-burst" bot.

Commands:
  ic2x            Run the loop (default; same as `ic2x bot`)
  ic2x bot        Fetch newest unseen burst → pick the best → post; every N hours
  ic2x login      Interactive iCloud sign-in (2FA) to establish the session
  ic2x status     Offline snapshot of bot state (last/next post, counts, today's spend)
  ic2x cost       Show estimated AI spend per day
  ic2x compare    Compare judge models on recent bursts (read-only, no posting)
  ic2x clean      Discard non-posted image records

Diagnostics (read-only, no posting): classify, autorotate, scene-dedup-test,
enhance-test, polish-test — each writes a *_out/<timestamp>/ folder to review.
"""

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="ic2x", description="Autonomous iCloud → X best-of-burst bot")
    sub = p.add_subparsers(dest="cmd")

    botp = sub.add_parser("bot", help="Run the autonomous loop (default)")
    botp.add_argument("--once", action="store_true",
                      help="run a single cycle and exit (for testing; forces a cycle, "
                           "ignores the post interval + daily cap)")
    botp.add_argument("--test", action="store_true",
                      help="fast speed-run soak: 1s waits, cycles until --test-cycles / library "
                           "exhausted / AI cap. Does NOT post to X (isolated throwaway state under "
                           "test_run/) unless --post is given. iCloud + AI calls are REAL (cost money).")
    botp.add_argument("--post", action="store_true",
                      help="with --test: post to X for REAL (uses the real state.db so posts are "
                           "tracked, no duplicates). Without it, --test never posts.")
    botp.add_argument("--test-cycles", type=int, default=20, metavar="N",
                      help="max cycles in --test mode (default 20; 0 = unlimited)")
    sub.add_parser("login", help="Interactive iCloud sign-in (2FA) to establish the session")
    cmp = sub.add_parser("compare", help="Compare judge models on recent bursts (read-only)")
    cmp.add_argument("--models", default="gemini-2.5-flash-lite; qwen3.5-flash",
                     help="';'-separated model ids (each may carry a ', effort' suffix, "
                          "e.g. 'qwen3.5-flash, off')")
    cmp.add_argument("--bursts", type=int, default=5, help="number of recent bursts to test")
    cmp.add_argument("--keep", action="store_true",
                     help="save evaluated thumbnails + verdicts.md to compare_out/ for review")
    cls = sub.add_parser("classify",
                         help="Judge N newest non-screenshot photos → classify_out/ (no posting)")
    cls.add_argument("--count", type=int, default=30, help="number of photos to classify")
    cls.add_argument("--rotate", action="store_true",
                     help="run the AI rotation pass on each image before judging")
    rot = sub.add_parser("autorotate",
                         help="Test the AI rotation pass on N newest photos → rotate_out/")
    rot.add_argument("--count", type=int, default=30, help="number of photos to rotate")
    rot.add_argument("--models", default="",
                     help="';'-separated rotation models to compare (default: ROTATION_MODEL)")
    sdt = sub.add_parser("scene-dedup-test",
                         help="Calibrate same-scene dedup on a folder of images (read-only)")
    sdt.add_argument("--dir", required=True,
                     help="folder of images; first (sorted) = candidate, rest = 'recent posts'")
    sdt.add_argument("--model", default="", help="override SCENE_DEDUP_MODEL for this test")
    enh = sub.add_parser("enhance-test",
                         help="Test Aliyun VIAPI image enhancement on N photos → enhance_out/")
    enh.add_argument("--count", type=int, default=8, help="number of photos to enhance")
    enh.add_argument("--dir", default="", help="folder of images (default: iCloud newest N)")
    enh.add_argument("--capabilities", default="superres,color",
                     help="comma list of: superres, color, color-strong")
    enh.add_argument("--max-edge", type=int, default=1920,
                     help="downscale long edge to this before sending (API max 1920)")
    pol = sub.add_parser("polish-test",
                         help="Preview the free local adaptive polish on N photos → polish_out/ (no cost)")
    pol.add_argument("--count", type=int, default=8, help="number of photos to polish")
    pol.add_argument("--dir", default="", help="folder of images (default: iCloud newest N)")
    pol.add_argument("--intensities", default="natural,punchy",
                     help="comma list of intensities to compare: natural, punchy")
    cst = sub.add_parser("cost", help="Show the bot's estimated AI spend per day")
    cst.add_argument("--days", type=int, default=14, help="how many recent days to show")
    sts = sub.add_parser("status",
                         help="Offline snapshot of bot state (last/next post, counts, today's spend)")
    sts.add_argument("--json", action="store_true", help="print the snapshot as JSON (for monitoring)")
    cln = sub.add_parser("clean", help="Discard non-posted image records")
    cln.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")

    args = p.parse_args()
    cmd = args.cmd or "bot"

    if cmd == "bot":
        from ic2x.bot import bot
        bot(once=getattr(args, "once", False),
            test=getattr(args, "test", False),
            test_cycles=getattr(args, "test_cycles", 20),
            post=getattr(args, "post", False))
    elif cmd == "login":
        from ic2x.login import login
        login()
    elif cmd == "compare":
        from ic2x.compare import compare
        compare(models=[m.strip() for m in args.models.split(";") if m.strip()],
                n_bursts=args.bursts, keep=args.keep)
    elif cmd == "classify":
        from ic2x.classify import classify
        classify(count=args.count, rotate=args.rotate)
    elif cmd == "autorotate":
        from ic2x.rotate import autorotate
        models = [m.strip() for m in args.models.split(";") if m.strip()] or None
        autorotate(count=args.count, models=models)
    elif cmd == "scene-dedup-test":
        from ic2x.scene_dedup_test import scene_dedup_test
        scene_dedup_test(folder=args.dir, model=args.model.strip() or None)
    elif cmd == "enhance-test":
        from ic2x.enhance_test import enhance_test
        caps = tuple(c.strip() for c in args.capabilities.split(",") if c.strip())
        enhance_test(count=args.count, capabilities=caps,
                     folder=args.dir.strip() or None, max_edge=args.max_edge)
    elif cmd == "polish-test":
        from ic2x.polish_test import polish_test
        its = tuple(i.strip() for i in args.intensities.split(",") if i.strip())
        polish_test(count=args.count, folder=args.dir.strip() or None, intensities=its)
    elif cmd == "cost":
        from ic2x.cost import cost
        cost(days=args.days)
    elif cmd == "status":
        from ic2x.overview import status
        status(as_json=getattr(args, "json", False))
    elif cmd == "clean":
        from ic2x.clean import clean
        clean(assume_yes=getattr(args, "yes", False))
